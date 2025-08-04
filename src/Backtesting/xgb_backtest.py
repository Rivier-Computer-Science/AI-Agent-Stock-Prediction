# File: src/Backtesting/xgb_backtest.py
"""
Backtest script for the XGB CNN-Attention-LSTM + XGBoost pipeline using CrewAI DecisionAgent.

Features:
- Data ingestion & windowing via the main pipeline.
- Pretrained model inference to get cnn_lstm_score and xgb_prob.
- DecisionAgent (CrewAI) mapping probability to BUY/SELL/HOLD.
- Simple strategy execution with position tracking, commissions, and PnL.
- Performance summary: cumulative return, Sharpe ratio, max drawdown.
- Start portfolio value in summary reflects the true initial capital.
"""

import os
import sys
import json
import re
from datetime import datetime

import numpy as np
import pandas as pd
import torch

# Ensure project root is on path so we can import the main XGB pipeline
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, HERE)

# Import the main pipeline (expects src/UI/xgb.py to define required functions/classes)
from src.UI import xgb as xgb_module  # main pipeline including DecisionAgent, inference, etc.

# Default backtest parameters
DEFAULT_CNN_LSTM_PATH = "src/UI/best_model.pt"
DEFAULT_XGB_PATH = "src/UI/xgb_model.joblib"
DEFAULT_INITIAL_CAPITAL = 10000.0
DEFAULT_COMMISSION_RATE = 0.001  # 0.1% per trade
DEFAULT_WINDOW_SIZE = 20


# ---------- Utility Functions ----------
def calculate_max_drawdown(equity_curve: pd.Series) -> dict:
    """
    Compute maximum drawdown and its duration.
    """
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    max_dd = drawdown.min()
    end = drawdown.idxmin()
    start_candidates = equity_curve[:end][equity_curve[:end] == roll_max[:end]]
    start = start_candidates.last_valid_index() if not start_candidates.empty else None
    duration = (end - start).days if start is not None else None
    return {
        "max_drawdown": float(max_dd),
        "drawdown_start": str(start) if start is not None else None,
        "drawdown_end": str(end),
        "duration_days": duration
    }


def first_level_number(idx: pd.MultiIndex, name: str):
    """
    Return the numeric level of `name` in a MultiIndex, handling duplicates by choosing the first match.
    """
    for i, n in enumerate(idx.names):
        if n == name:
            return i
    raise ValueError(f"Level name {name} not found in index.names")


def _extract_json_object(raw: str) -> str:
    """
    Extract JSON object from potentially fenced or noisy LLM output.
    """
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    m = re.search(r"\{[\s\S]*\}", s)
    return m.group(0) if m else s


# ---------- Signal Generation ----------
def generate_signals(
    ticker: str,
    start: str,
    end: str,
    cnn_lstm_path: str,
    xgb_path: str,
    window_size: int = DEFAULT_WINDOW_SIZE,
    device: torch.device = None
) -> pd.DataFrame:
    """
    Run the pretrained pipeline over historical data to produce per-window BUY/SELL/HOLD signals.
    Uses CrewAI DecisionAgent for recommendation.
    Returns DataFrame indexed by date with columns: cnn_lstm_score, xgb_prob, recommendation.
    """
    print(f"[XGB.Backtest] Generating signals for {ticker} from {start} to {end}")

    # 1. Data ingestion and window slicing
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = xgb_module.fetch_ohlcv([ticker], start, end)
    df = xgb_module.preprocess_data(df)
    df = xgb_module.normalize_features(df, features)
    X, y = xgb_module.slice_windows(df, features, window_size)
    if len(X) == 0:
        raise RuntimeError("Not enough data to form any window for backtest.")

    # 2. Align windows to decision dates robustly with MultiIndex handling
    ticker_lvl = first_level_number(df.index, 'Ticker')
    date_lvl = first_level_number(df.index, 'Date')
    ticker_mask = df.index.get_level_values(ticker_lvl) == ticker
    dates_raw = df.index.get_level_values(date_lvl)[ticker_mask]
    dates = pd.DatetimeIndex(dates_raw).sort_values().unique()
    expected_windows = len(dates) - window_size
    if expected_windows <= 0:
        raise RuntimeError("Not enough time steps to form windows with the given window_size.")

    if len(X) != expected_windows:
        min_len = min(len(X), expected_windows)
        X = X[:min_len]
        decision_dates = dates[window_size: window_size + min_len]
    else:
        decision_dates = dates[window_size: window_size + len(X)]

    # 3. Load pretrained models
    cnn_lstm, xgb_model, device = xgb_module.load_models(
        cnn_lstm_path=cnn_lstm_path,
        xgb_path=xgb_path,
        device=device
    )

    # 4. DecisionAgent instantiation
    decision_agent = xgb_module.DecisionAgent()

    # 5. Iterate windows, perform inference, and get CrewAI decisions
    records = []
    for window, decision_date in zip(X, decision_dates):
        scores = xgb_module.infer_window(cnn_lstm, xgb_model, window, scaler=None, device=device)
        inference_json = xgb_module.make_inference_json(decision_date.strftime('%Y-%m-%d'), scores)

        try:
            recommendation = decision_agent.get_decision(inference_json)
        except Exception as e:
            # Fallback parsing: directly call LLM and extract JSON
            print(f"[XGB.Backtest] DecisionAgent primary call failed on {decision_date}: {e}")
            prompt = decision_agent.create_prompt(inference_json)
            raw = xgb_module.call_llm(decision_agent.llm, prompt)
            cleaned = _extract_json_object(raw)
            try:
                recommendation = json.loads(cleaned)
                print(f"[XGB.Backtest] Fallback parsed recommendation: {recommendation}")
            except Exception as e2:
                print(f"[XGB.Backtest] Fallback parse failed for {decision_date}. Defaulting to HOLD. raw: {raw}, cleaned: {cleaned}, error: {e2}")
                recommendation = {"recommendation": "HOLD"}

        rec = recommendation.get("recommendation", "HOLD").upper()
        records.append({
            "date": decision_date,
            "cnn_lstm_score": scores["cnn_lstm_score"],
            "xgb_prob": scores["xgb_prob"],
            "recommendation": rec
        })

    signal_df = pd.DataFrame(records).set_index("date").sort_index()
    print(f"[XGB.Backtest] Generated {len(signal_df)} signals.")
    return signal_df


# ---------- Backtest Execution ----------
def run_simple_backtest(
    signals: pd.DataFrame,
    price_df: pd.DataFrame,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    commission_rate: float = DEFAULT_COMMISSION_RATE
) -> (pd.DataFrame, pd.DataFrame):
    """
    Execute a simple trading simulation based on BUY/SELL/HOLD signals.
    Trades occur at close price. Positions are exited/flipped per signal change, with commission.
    """
    df = signals.copy()
    price_close = price_df['Close'].copy()
    df = df.join(price_close.rename("close"), how='left')
    df = df.dropna(subset=["close"])
    dates = df.index.sort_values()

    position = 0  # +1 long, -1 short, 0 flat
    cash = initial_capital
    prev_price = None
    equity_curve = []
    trade_log = []

    for dt in dates:
        rec = df.loc[dt, "recommendation"]
        price = df.loc[dt, "close"]

        target_pos = 0
        if rec == "BUY":
            target_pos = 1
        elif rec == "SELL":
            target_pos = -1
        elif rec == "HOLD":
            target_pos = position  # maintain

        # Position change triggers entry/exit/flip and fees
        if target_pos != position:
            if position != 0:
                fee = abs(position) * price * commission_rate
                cash -= fee
                trade_log.append({
                    "date": dt,
                    "action": "EXIT" if target_pos == 0 else "FLIP",
                    "from": position,
                    "to": target_pos,
                    "price": price,
                    "fee": fee
                })
            if target_pos != 0:
                fee = abs(target_pos) * price * commission_rate
                cash -= fee
                trade_log.append({
                    "date": dt,
                    "action": "ENTER",
                    "from": position,
                    "to": target_pos,
                    "price": price,
                    "fee": fee
                })
            position = target_pos

        # PnL from price movement
        if prev_price is not None and position != 0:
            pnl = position * (price - prev_price)
        else:
            pnl = 0.0

        portfolio_value = cash + position * price + pnl
        equity_curve.append({
            "date": dt,
            "position": position,
            "price": price,
            "cash": cash,
            "pnl": pnl,
            "total_value": portfolio_value
        })

        prev_price = price

    equity_df = pd.DataFrame(equity_curve).set_index("date")
    equity_df["total_value"] = equity_df["total_value"].ffill()
    equity_df["daily_return"] = equity_df["total_value"].pct_change().fillna(0.0)
    return equity_df, pd.DataFrame(trade_log)


def summarize_performance(equity_df: pd.DataFrame, initial_capital: float) -> dict:
    """
    Compute performance metrics using provided initial capital (so start_value is exact).
    """
    returns = equity_df["daily_return"]
    cumulative_return = equity_df["total_value"].iloc[-1] / initial_capital - 1
    mean_ret = returns.mean()
    std_ret = returns.std(ddof=0)
    sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else None
    mdd = calculate_max_drawdown(equity_df["total_value"])
    summary = {
        "cumulative_return": float(cumulative_return),
        "sharpe_ratio": float(sharpe) if sharpe is not None else None,
        "max_drawdown": mdd["max_drawdown"],
        "drawdown_duration_days": mdd["duration_days"],
        "start_value": float(initial_capital),
        "end_value": float(equity_df["total_value"].iloc[-1])
    }
    return summary


def run_backtest(
    ticker: str,
    start: str,
    end: str,
    cnn_lstm_path: str,
    xgb_path: str,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    commission_rate: float = DEFAULT_COMMISSION_RATE,
    window_size: int = DEFAULT_WINDOW_SIZE
):
    """
    Full backtest entrypoint orchestrating signal generation, simulation, and summarization.
    """
    print(f"[XGB.Backtest] Running backtest for {ticker} from {start} to {end}")

    # 1. Signal generation
    signals = generate_signals(
        ticker=ticker,
        start=start,
        end=end,
        cnn_lstm_path=cnn_lstm_path,
        xgb_path=xgb_path,
        window_size=window_size
    )

    # 2. Fetch price series for PnL
    ohlcv = xgb_module.fetch_ohlcv([ticker], start, end)
    try:
        price_df = ohlcv.loc[ticker]
    except Exception:
        price_df = ohlcv.xs(ticker, level=0)
    price_df = price_df.sort_index()
    if not isinstance(price_df.index, pd.DatetimeIndex):
        price_df.index = pd.to_datetime(price_df.index)

    # 3. Execute trading simulation
    equity_df, trades = run_simple_backtest(
        signals=signals,
        price_df=price_df,
        initial_capital=initial_capital,
        commission_rate=commission_rate
    )

    # 4. Performance summary (use initial capital to get canonical start_value)
    perf = summarize_performance(equity_df, initial_capital)
    print("===== Backtest Summary =====")
    print(f"Cumulative Return: {perf['cumulative_return'] * 100:.2f}%")
    if perf['sharpe_ratio'] is not None:
        print(f"Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
    else:
        print("Sharpe Ratio: N/A")
    print(f"Max Drawdown: {perf['max_drawdown'] * 100:.2f}% over {perf['drawdown_duration_days']} days")
    print(f"Start Portfolio Value: {perf['start_value']:.2f}")
    print(f"End Portfolio Value: {perf['end_value']:.2f}")

    return {
        "signals": signals,
        "equity_curve": equity_df,
        "trades": trades,
        "performance": perf
    }


# ---------- CLI Entry Point ----------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Backtest XGB CNN-Attention-LSTM + XGBoost pipeline with CrewAI DecisionAgent"
    )
    parser.add_argument("--ticker", help="Ticker symbol, e.g., AAPL")
    parser.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD")
    parser.add_argument(
        "--cnn-lstm", default=DEFAULT_CNN_LSTM_PATH,
        help=f"Path to pretrained CNN+Attention+LSTM model (default: {DEFAULT_CNN_LSTM_PATH})"
    )
    parser.add_argument(
        "--xgb", default=DEFAULT_XGB_PATH,
        help=f"Path to trained XGBoost model (default: {DEFAULT_XGB_PATH})"
    )
    parser.add_argument(
        "--capital", type=float, default=DEFAULT_INITIAL_CAPITAL,
        help=f"Initial capital (default: {DEFAULT_INITIAL_CAPITAL})"
    )
    parser.add_argument(
        "--commission", type=float, default=DEFAULT_COMMISSION_RATE,
        help=f"Commission rate per trade (fraction) (default: {DEFAULT_COMMISSION_RATE})"
    )
    parser.add_argument(
        "--window", type=int, default=DEFAULT_WINDOW_SIZE,
        help=f"Window size for inference (default: {DEFAULT_WINDOW_SIZE})"
    )
    args = parser.parse_args()

    # Prompt user if required arguments are missing
    ticker = args.ticker or input("Enter stock ticker (e.g., AAPL): ").strip().upper()
    start = args.start or input("Enter start date (YYYY-MM-DD): ").strip()
    end = args.end or input("Enter end date (YYYY-MM-DD): ").strip()

    # Validate dates
    try:
        pd.to_datetime(start)
        pd.to_datetime(end)
    except Exception:
        print("Invalid date format. Use YYYY-MM-DD.")
        sys.exit(1)

    result = run_backtest(
        ticker=ticker,
        start=start,
        end=end,
        cnn_lstm_path=args.cnn_lstm,
        xgb_path=args.xgb,
        initial_capital=args.capital,
        commission_rate=args.commission,
        window_size=args.window
    )

    # Persist outputs
    out_dir = "backtest_results"
    os.makedirs(out_dir, exist_ok=True)
    result["signals"].to_csv(os.path.join(out_dir, f"{ticker}_signals.csv"))
    result["equity_curve"].to_csv(os.path.join(out_dir, f"{ticker}_equity.csv"))
    result["trades"].to_csv(os.path.join(out_dir, f"{ticker}_trades.csv"))
    with open(os.path.join(out_dir, f"{ticker}_performance.json"), "w") as f:
        json.dump(result["performance"], f, indent=2)
    print(f"[XGB.Backtest] Results written to {out_dir}")
