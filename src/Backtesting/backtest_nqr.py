# File: src/Backtesting/nqr_backtesting.py
"""
Backtest script for the NQR pipeline (src/UI/nqr.py) with:
  - Correct, robust Crew AI (OpenAI) call path (new + legacy SDKs), retries, strict JSON parsing,
    and a local majority-vote fallback if all else fails.
  - Daily signals via simple technical features (momentum, RSI, volatility) so LLM hiccups
    don’t wipe out the entire test period.

Notes:
- This uses the NQR demo models (tiny FNN + RF on synthetic labels) only to produce
  probabilities; it's *not* predictive finance. The point is to exercise the pipeline.
- No files are written. Summary prints to stdout.
"""

import os
import sys
import json
import time
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
import yfinance as yf

# Ensure project root on path so we can import the NQR pipeline module
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Import the main NQR pipeline (src/UI/nqr.py)
from src.UI import nqr as nqr_module


# =============================================================================
# LLM (Crew AI) call path — robust wrapper
# =============================================================================

def _extract_json_object(raw: str) -> str:
    """
    Extract JSON object from a possibly fenced/noisy LLM output.
    - Strips code fences if present.
    - Returns substring between outermost {...} if found; else returns original.
    """
    s = str(raw or "").strip()
    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 3:
            s = parts[1].strip()
        else:
            s = s.replace("```", "")
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start:end + 1]
    return s


def _majority_vote_fallback(fnn_prob: float, rf_prob: float, anfis_prob: Optional[float], threshold: float) -> Dict[str, str]:
    """
    Last-resort deterministic decision if LLM call fails repeatedly.
    """
    probs = [fnn_prob, rf_prob] + ([anfis_prob] if anfis_prob is not None else [])
    buys = sum(p > threshold for p in probs)
    sells = sum(p < threshold for p in probs)
    if buys >= 2:
        decision = "BUY"
    elif sells >= 2:
        decision = "SELL"
    else:
        decision = "HOLD"
    return {
        "decision": decision,
        "explanation": f"Local majority vote with threshold={threshold}. Probs={probs}"
    }


def call_crewai_decider(
    fnn_prob: float,
    rf_prob: float,
    anfis_prob: Optional[float],
    threshold: float = 0.5,
    model_name: str = "gpt-4o",
    max_retries: int = 4,
    backoff_base: float = 0.7
) -> Dict[str, str]:
    """
    Robust Crew AI decision call supporting both new and legacy OpenAI SDKs.

    Guarantees:
      - Returns a dict with keys {"decision", "explanation"}.
      - If API fails or returns non-JSON, retries with backoff.
      - If still failing, falls back to deterministic majority vote.

    Requires:
      - OPENAI_API_KEY in environment.
      - Model availability on your account (or adjust model_name).
    """
    system_text = (
        "You are a Crew AI agent specialized in combining multiple model probabilities "
        "into a single buy/hold/sell decision.\n"
        "Role: financial decision-maker\n"
        "Goal: Apply majority-vote or threshold rules to decide trade action."
    )
    user_text = (
        f"You are given the following model probabilities:\n"
        f"- FNN: {fnn_prob}\n"
        f"- RF: {rf_prob}\n"
        f"- ANFIS: {anfis_prob if anfis_prob is not None else 'N/A'}\n\n"
        f"Use threshold = {threshold}.\n"
        f"Count each probability > threshold as a BUY signal, < threshold as a SELL signal.\n"
        f"Apply majority vote:\n"
        f"- If 2 or more BUY signals → decision = BUY\n"
        f"- If 2 or more SELL signals → decision = SELL\n"
        f"- Otherwise → decision = HOLD\n\n"
        f"Respond in strict JSON format ONLY:\n"
        f'{{"decision":"BUY|SELL|HOLD","explanation":"short reasoning"}}'
    )

    # Try new SDK first
    client = None
    using_new_sdk = False
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI()
        using_new_sdk = True
    except Exception:
        # Fallback to legacy SDK
        try:
            import openai  # type: ignore
            # API key picked up from env variable
            client = openai
        except Exception:
            # No SDK available — go majority vote
            return _majority_vote_fallback(fnn_prob, rf_prob, anfis_prob, threshold)

    for attempt in range(max_retries):
        try:
            if using_new_sdk:
                # New SDK: enforce JSON with response_format
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_text},
                        {"role": "user", "content": user_text},
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},  # HARD JSON
                )
                content = resp.choices[0].message.content
            else:
                # Legacy SDK
                resp = client.ChatCompletion.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_text},
                        {"role": "user", "content": user_text},
                    ],
                    temperature=0.0,
                )
                content = resp.choices[0].message.content

            # Strict JSON parse
            try:
                payload = json.loads(content)
            except Exception:
                payload = json.loads(_extract_json_object(content))

            # Validate schema
            decision = str(payload.get("decision", "")).strip().upper()
            explanation = str(payload.get("explanation", "")).strip()
            if decision not in {"BUY", "SELL", "HOLD"}:
                raise ValueError(f"Invalid decision token: {decision}")

            return {"decision": decision, "explanation": explanation}

        except Exception as e:
            # Exponential backoff
            if attempt < max_retries - 1:
                time.sleep(backoff_base * (2 ** attempt))
            else:
                # Final fallback
                return _majority_vote_fallback(fnn_prob, rf_prob, anfis_prob, threshold)


# =============================================================================
# Technical features for *daily* signals
# =============================================================================

def _tanh01(x: pd.Series, scale: float = 3.0) -> pd.Series:
    """
    Squash a series to [0,1] using tanh for bounded, robust mapping.
    y = (tanh(scale * x) + 1) / 2
    """
    return (np.tanh(scale * x.astype(float)) + 1.0) / 2.0


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Classic RSI implementation. Returns values in [0, 100].
    """
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)

    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()

    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0).clip(0, 100)


def make_daily_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Build a 3-feature daily dataframe (index=trading days):
      feat1: 5-day momentum (tanh01 scaled)
      feat2: RSI-14 normalized to [0,1]
      feat3: 10-day realized volatility (scaled & clipped to [0,1])
    """
    s = prices["close"].astype(float)
    ret = s.pct_change()

    # Momentum over 5 days
    mom5 = s.pct_change(5).fillna(0.0)
    feat1 = _tanh01(mom5, scale=3.0)

    # RSI normalized to [0,1]
    rsi14 = _compute_rsi(s, period=14) / 100.0
    feat2 = rsi14.clip(0, 1)

    # Volatility (std of returns over 10 days), clipped around a cap for normalization
    vol10 = ret.rolling(10).std().fillna(0.0)
    vol_cap = 0.08  # ~8% daily std cap for scaling; tweak as desired
    feat3 = (vol10 / vol_cap).clip(0, 1)

    feats = pd.DataFrame(
        {"feat1": feat1, "feat2": feat2, "feat3": feat3},
        index=prices.index
    ).dropna()
    return feats


# =============================================================================
# Model training & inference (demo)
# =============================================================================

def train_demo_models(input_dim: int = 3, seed: int = 42):
    """
    Train the small FNN and RF (and ANFIS if available) on synthetic data.

    Even though features are technical daily signals, we still train on synthetic labels,
    purely to produce probabilities for the LLM combiner.
    """
    rng = np.random.default_rng(seed)
    X_demo = rng.normal(size=(512, input_dim)).astype(np.float32)
    y_demo = (rng.random(size=(512,)) > 0.5).astype(int)

    trainer = nqr_module.ModelTrain()
    fnn_model, _ = trainer.train_fnn(X_demo, y_demo, epochs=10, lr=0.003)
    rf_model, _ = trainer.train_rf(X_demo, y_demo)

    anfis_model = None
    if nqr_module.ANFISNet:
        try:
            anfis_model, _ = trainer.train_anfis(X_demo, y_demo)
        except Exception:
            anfis_model = None
    return fnn_model, rf_model, anfis_model


def infer_probs_from_models(
    fnn_model: torch.nn.Module,
    rf_model,
    features_row: np.ndarray,
    anfis_model=None,
) -> Dict[str, Optional[float]]:
    """
    Compute probabilities from in-memory models for a single daily feature row (shape=(3,)).
    """
    x = features_row.reshape(1, -1).astype(np.float32)
    with torch.no_grad():
        fnn_prob = float(fnn_model(torch.tensor(x, dtype=torch.float32)).item())
    rf_prob = float(rf_model.predict_proba(x)[0, 1])
    anfis_prob = None
    if anfis_model is not None:
        try:
            anfis_prob = float(anfis_model.predict(x)[0])
        except Exception:
            anfis_prob = None
    return {"fnn_prob": fnn_prob, "rf_prob": rf_prob, "anfis_prob": anfis_prob}


# =============================================================================
# Price fetching (robust)
# =============================================================================

def _extract_single_close(df: pd.DataFrame, ticker: str) -> pd.Series:
    """
    Robustly pull a single 'Close' Series for `ticker` from a yfinance DataFrame.
    Handles both single-index and MultiIndex columns.
    """
    if "Close" in df.columns and not isinstance(df.columns, pd.MultiIndex):
        return pd.to_numeric(df["Close"], errors="coerce")

    if isinstance(df.columns, pd.MultiIndex):
        if ("Close", ticker) in df.columns:
            return pd.to_numeric(df[("Close", ticker)], errors="coerce")
        if (ticker, "Close") in df.columns:
            return pd.to_numeric(df[(ticker, "Close")], errors="coerce")
        try:
            if "Close" in df.columns.get_level_values(0):
                s = df.xs("Close", axis=1, level=0, drop_level=True)
                if isinstance(s, pd.DataFrame):
                    s = s[ticker] if ticker in s.columns else s.iloc[:, 0]
                return pd.to_numeric(s, errors="coerce")
        except Exception:
            pass
        try:
            if "Close" in df.columns.get_level_values(-1):
                s = df.xs("Close", axis=1, level=-1, drop_level=True)
                if isinstance(s, pd.DataFrame):
                    s = s[ticker] if ticker in s.columns else s.iloc[:, 0]
                return pd.to_numeric(s, errors="coerce")
        except Exception:
            pass

    if "Adj Close" in df.columns and not isinstance(df.columns, pd.MultiIndex):
        return pd.to_numeric(df["Adj Close"], errors="coerce")

    if isinstance(df.columns, pd.MultiIndex):
        if ("Adj Close", ticker) in df.columns:
            return pd.to_numeric(df[("Adj Close", ticker)], errors="coerce")
        if (ticker, "Adj Close") in df.columns:
            return pd.to_numeric(df[(ticker, "Adj Close")], errors="coerce")

    raise RuntimeError("Could not find a Close series for the ticker in the downloaded DataFrame.")


def fetch_price_series(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch daily OHLCV for [start, end]; return DataFrame with a single numeric column 'close'.
    """
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"No price data for {ticker} in [{start}, {end}]")

    df = df.sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    close_series = _extract_single_close(df, ticker).dropna()
    if close_series.empty:
        raise RuntimeError(f"Close price series for {ticker} is empty after cleaning.")

    return close_series.to_frame(name="close")


# =============================================================================
# Daily signal generation (LLM calls per day; trade next day)
# =============================================================================

def generate_daily_signals(
    ticker: str,
    start: str,
    end: str,
    threshold: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create **daily** BUY/SELL/HOLD signals:
      - Build 3 daily features (feat1/feat2/feat3).
      - Train demo models on synthetic labels (once).
      - For each trading day t (after warmup/NaN drop):
          -> get model probabilities on features[t]
          -> call Crew AI decider with retries + strict JSON
          -> store 'decision' for day t
      - To avoid look-ahead, we will EXECUTE the action on day t+1 in the backtest.

    Returns
    -------
    DataFrame indexed by date with columns:
      [feat1, feat2, feat3, fnn_prob, rf_prob, anfis_prob, decision, explanation]
    """
    prices = fetch_price_series(ticker, start, end)
    feats = make_daily_features(prices)
    if feats.empty:
        raise RuntimeError("Not enough data to compute daily features.")

    # Train demo models (input_dim=3 to match feat1/2/3)
    fnn_model, rf_model, anfis_model = train_demo_models(input_dim=3, seed=seed)

    rows = []
    for dt, row in feats.iterrows():
        features_vec = row.values.astype(np.float32)  # [feat1, feat2, feat3]
        probs = infer_probs_from_models(fnn_model, rf_model, features_vec, anfis_model)
        # Robust Crew AI call (new SDK if available; strict JSON; retries; fallback to majority vote)
        result = call_crewai_decider(
            fnn_prob=probs["fnn_prob"],
            rf_prob=probs["rf_prob"],
            anfis_prob=probs["anfis_prob"],
            threshold=threshold,
            model_name="gpt-4o",
            max_retries=4,
        )
        rows.append({
            "date": pd.Timestamp(dt),
            "feat1": float(row["feat1"]),
            "feat2": float(row["feat2"]),
            "feat3": float(row["feat3"]),
            "fnn_prob": probs["fnn_prob"],
            "rf_prob": probs["rf_prob"],
            "anfis_prob": probs["anfis_prob"],
            "decision": str(result.get("decision", "HOLD")).upper(),
            "explanation": str(result.get("explanation", "")),
        })

    sig = pd.DataFrame(rows).set_index("date").sort_index()
    # Normalize to {BUY, SELL, HOLD}
    sig["decision"] = (
        sig["decision"].astype(str).str.upper().str.strip()
        .replace({"": "HOLD", "N/A": "HOLD", "NONE": "HOLD", "NULL": "HOLD"})
    )
    sig.loc[~sig["decision"].isin(["BUY", "SELL", "HOLD"]), "decision"] = "HOLD"
    return sig


def align_to_next_trading_day(signals: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    Map each daily signal date to the *next* trading day to avoid look-ahead.
    Since signals are already daily and aligned to price days, this is effectively a +1 shift.
    We still handle missing next-day (e.g. last day of range) by dropping those rows.
    """
    idx = prices.index
    exec_dates = []
    for dt in signals.index:
        pos = idx.searchsorted(dt) + 1  # next day
        exec_dates.append(idx[pos] if pos < len(idx) else pd.NaT)

    out = signals.copy()
    out["exec_date"] = exec_dates
    out = out.dropna(subset=["exec_date"]).copy()
    out["exec_date"] = pd.to_datetime(out["exec_date"])
    out = out.set_index("exec_date", drop=False).sort_index()
    # If multiple signals map to the same exec date (unlikely with daily), keep last
    out = out[~out.index.duplicated(keep="last")]
    return out


# =============================================================================
# Backtest core
# =============================================================================

def run_simple_backtest(
    aligned_signals: pd.DataFrame,
    prices: pd.DataFrame,
    initial_capital: float = 10_000.0,
    commission_rate: float = 0.001,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Daily long/short backtest:
      - BUY  => +1
      - SELL => -1
      - HOLD =>  0
    Trades executed at close on each trading day when target changes.
    """
    prices = prices.copy().sort_index()

    # Map decisions to {-1,0,1}; carry forward to ensure a position persists
    dec = aligned_signals["decision"].astype(str).str.upper().str.strip()
    dec = dec.where(dec.isin(["BUY", "SELL", "HOLD"]), other="HOLD")
    pos_map = {"BUY": 1, "SELL": -1, "HOLD": 0}
    daily_target = dec.map(pos_map).reindex(prices.index).ffill().fillna(0).astype(int)

    close_series = prices["close"].astype(float)

    position = 0
    cash = float(initial_capital)
    prev_price = None

    equity_rows = []
    trades = []

    for dt, price_val in close_series.items():
        price = float(price_val)

        # Change position at today's close if needed
        target = int(daily_target.loc[dt])
        if target != position:
            fee = abs(target - position) * price * commission_rate
            cash -= fee
            action = "ENTER" if position == 0 and target != 0 else ("EXIT" if target == 0 else "FLIP")
            trades.append({
                "date": dt,
                "action": action,
                "from": position,
                "to": target,
                "price": price,
                "fee": fee,
            })
            position = target

        # PnL from price move with current position
        pnl = 0.0
        if prev_price is not None and position != 0:
            pnl = position * (price - prev_price)

        total_value = cash + position * price + pnl
        equity_rows.append({
            "date": dt,
            "position": position,
            "price": price,
            "cash": cash,
            "pnl": pnl,
            "total_value": total_value,
        })
        prev_price = price

    equity_df = pd.DataFrame(equity_rows).set_index("date")
    equity_df["total_value"] = equity_df["total_value"].ffill()
    equity_df["daily_return"] = equity_df["total_value"].pct_change().fillna(0.0)

    trades_df = pd.DataFrame(trades)
    return equity_df, trades_df


def summarize_performance(equity_df: pd.DataFrame, initial_capital: float) -> Dict[str, Optional[float]]:
    """
    Simple performance summary for convenience.
    """
    if equity_df.empty:
        return {
            "cumulative_return": 0.0,
            "sharpe_ratio": None,
            "max_drawdown": 0.0,
            "drawdown_duration_days": None,
            "start_value": float(initial_capital),
            "end_value": float(initial_capital),
        }

    ret = equity_df["daily_return"]
    cum_ret = equity_df["total_value"].iloc[-1] / initial_capital - 1
    std = ret.std(ddof=0)
    sharpe = (ret.mean() / std) * np.sqrt(252) if std and std > 0 else None

    # Max DD
    roll_max = equity_df["total_value"].cummax()
    dd = (equity_df["total_value"] - roll_max) / roll_max
    max_dd = float(dd.min())
    end = dd.idxmin()
    start_candidates = equity_df["total_value"][:end][equity_df["total_value"][:end] == roll_max[:end]]
    start = start_candidates.last_valid_index() if not start_candidates.empty else None
    duration = (end - start).days if start is not None else None

    return {
        "cumulative_return": float(cum_ret),
        "sharpe_ratio": float(sharpe) if sharpe is not None else None,
        "max_drawdown": max_dd,
        "drawdown_duration_days": duration,
        "start_value": float(initial_capital),
        "end_value": float(equity_df["total_value"].iloc[-1]),
    }


# =============================================================================
# Orchestrator
# =============================================================================

def run_backtest(
    ticker: str,
    start: str,
    end: str,
    initial_capital: float = 10_000.0,
    commission_rate: float = 0.001,
    threshold: float = 0.5,
    seed: int = 42,
) -> Dict[str, object]:
    """
    End-to-end backtest with daily signals and robust Crew AI path.
      1) Build daily features & probabilities.
      2) LLM decision with retries/JSON enforcement; fallback to local majority vote.
      3) Execute trades next day; compute equity curve and summary.
    """
    print(f"[NQR.Backtest] Creating daily signals for {ticker} in [{start}, {end}]...")
    daily_signals = generate_daily_signals(
        ticker=ticker,
        start=start,
        end=end,
        threshold=threshold,
        seed=seed,
    )

    prices = fetch_price_series(ticker, start, end)
    print("[NQR.Backtest] Aligning decisions to *next* trading day (no look-ahead)...")
    aligned = align_to_next_trading_day(daily_signals, prices)

    print("[NQR.Backtest] Running daily long/short backtest...")
    equity_df, trades_df = run_simple_backtest(
        aligned_signals=aligned,
        prices=prices,
        initial_capital=initial_capital,
        commission_rate=commission_rate,
    )

    perf = summarize_performance(equity_df, initial_capital)
    print("===== NQR Backtest Summary =====")
    print(f"Cumulative Return: {perf['cumulative_return'] * 100:.2f}%")
    print(f"Sharpe Ratio: {perf['sharpe_ratio']:.2f}" if perf['sharpe_ratio'] is not None else "Sharpe Ratio: N/A")
    print(f"Max Drawdown: {perf['max_drawdown'] * 100:.2f}% over {perf['drawdown_duration_days']} days")
    print(f"Start Portfolio Value: {perf['start_value']:.2f}")
    print(f"End Portfolio Value: {perf['end_value']:.2f}")

    return {
        "signals_daily": daily_signals,      # decisions for day t
        "signals_aligned": aligned,          # executed on day t+1
        "equity_curve": equity_df,
        "trades": trades_df,
        "performance": perf,
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="NQR daily backtest with robust Crew AI (OpenAI) calls and strict JSON."
    )
    parser.add_argument("--ticker", help="Ticker symbol, e.g., AAPL")
    parser.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD")
    parser.add_argument("--capital", type=float, default=10_000.0, help="Initial capital (default: 10000.0)")
    parser.add_argument("--commission", type=float, default=0.001, help="Commission rate per trade (default: 0.001)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold passed to CrewAI decider")
    parser.add_argument("--seed", type=int, default=42, help="Seed for synthetic training data")

    args = parser.parse_args()

    # Prompt for missing args
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

    _ = run_backtest(
        ticker=ticker,
        start=start,
        end=end,
        initial_capital=args.capital,
        commission_rate=args.commission,
        threshold=args.threshold,
        seed=args.seed,
    )
