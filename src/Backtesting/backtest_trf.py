import os
import sys
import logging
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import joblib
import yfinance as yf

# Keep third-party libraries quiet so the console only shows the backtest summary
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
for name in ("litellm", "LiteLLM", "openai", "httpx", "urllib3", "langchain", "langsmith"):
    logging.getLogger(name).setLevel(logging.ERROR)
logging.basicConfig(level=logging.WARNING, format="")

# Resolve project root & default paths for trained artifacts
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

UI_DIR = os.path.join(HERE, "src", "UI")
DEFAULT_TRANSFORMER_PATH = os.path.join(UI_DIR, "transformer.pt")
DEFAULT_XGB_PATH = os.path.join(UI_DIR, "xgb_model.joblib")

# Import core TRF modules trained in your UI pipeline
from src.UI.trf import DataHandler, TransformerModel  # noqa: E402

# CrewAI bits
from crewai import Agent, Task, Crew, Process  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402


# -----------------------------------------------------------------------------
# CrewAI decision agent wrapper with a two-step prompt strategy
# -----------------------------------------------------------------------------
class ForceActionCrewAgent:
    """
    CrewAI wrapper that queries the model twice:
      1) First pass: allow HOLD per strict rules.
      2) Second pass: if HOLD, ask again with BUY/SELL only.
    """

    # Initialize the CrewAI agent, model name, and neutral band thresholds.
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.0,
                 buy_hi: float = 0.525, sell_lo: float = 0.475, verbose: bool = False):
        # Require an OpenAI key since we do not implement a local fallback here
        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY not set for CrewAI.")
        self.buy_hi = float(buy_hi)
        self.sell_lo = float(sell_lo)
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        # Single agent; we run tasks sequentially
        self.agent = Agent(
            role="TRF Backtest Decision Agent",
            goal=("Convert xgb_prob and simple price context into decisive BUY/SELL/HOLD."),
            backstory="You assist a backtest; neutrality band is narrow.",
            verbose=verbose,
            allow_delegation=False,
            llm=self.llm,
        )

    # Build first-pass prompt that allows HOLD when rules say so.
    def _prompt_allow_hold(self, ticker: str, xgb_prob: float, trend: str, momentum: str) -> str:
        return (
            "Decide BUY/SELL/HOLD for a trading backtest.\n"
            f"Ticker: {ticker}\n"
            f"xgb_prob (P(up)): {xgb_prob:.6f}\n"
            f"trend: {trend} (UP/DOWN/FLAT)\n"
            f"momentum (10SMA vs 30SMA): {momentum} (FAST_ABOVE/FAST_BELOW/UNDEFINED)\n"
            "Rules:\n"
            f"  • If xgb_prob >= {self.buy_hi:.3f} → BUY\n"
            f"  • If xgb_prob <= {self.sell_lo:.3f} → SELL\n"
            "  • Else (neutral band):\n"
            "       – If trend==UP or momentum==FAST_ABOVE → BUY\n"
            "       – If trend==DOWN or momentum==FAST_BELOW → SELL\n"
            "       – Else HOLD\n"
            'Output exactly one JSON line: {"recommendation":"BUY|SELL|HOLD"}'
        )

    # Build second-pass prompt that forbids HOLD and forces BUY/SELL.
    def _prompt_force(self, ticker: str, xgb_prob: float, trend: str, momentum: str) -> str:
        return (
            "You MUST choose BUY or SELL (HOLD is forbidden).\n"
            f"Ticker: {ticker}\n"
            f"xgb_prob: {xgb_prob:.6f}\n"
            f"trend: {trend}\n"
            f"momentum: {momentum}\n"
            "Guideline: pick BUY if evidence leans up, otherwise SELL.\n"
            'Output exactly: {"recommendation":"BUY"} or {"recommendation":"SELL"}'
        )

    # Extract and validate {"recommendation": "..."} JSON from an LLM string response.
    @staticmethod
    def _extract_json(text: str) -> str:
        s = text.strip()
        i = s.find("{"); j = s.find("}", i)
        if i == -1 or j == -1:
            raise RuntimeError(f"CrewAI did not return JSON: {text!r}")
        import json
        obj = json.loads(s[i:j+1])
        rec = str(obj.get("recommendation", "")).upper()
        if rec not in {"BUY", "SELL", "HOLD"}:
            raise RuntimeError(f"Invalid recommendation: {obj!r}")
        return rec

    # Run the two-pass decision flow and return a final BUY or SELL.
    def decide(self, ticker: str, xgb_prob: float, trend: str, momentum: str) -> str:
        # First pass (may return HOLD)
        task1 = Task(
            description=self._prompt_allow_hold(ticker, xgb_prob, trend, momentum),
            expected_output='{"recommendation":"BUY|SELL|HOLD"}',
            agent=self.agent,
        )
        res1 = Crew(agents=[self.agent], tasks=[task1], process=Process.sequential).kickoff()
        rec1 = self._extract_json(str(res1))
        if rec1 != "HOLD":
            return rec1  # Early exit if we already got action
        # Second pass (HOLD not allowed)
        task2 = Task(
            description=self._prompt_force(ticker, xgb_prob, trend, momentum),
            expected_output='{"recommendation":"BUY|SELL"}',
            agent=self.agent,
        )
        res2 = Crew(agents=[self.agent], tasks=[task2], process=Process.sequential).kickoff()
        rec2 = self._extract_json(str(res2))
        # As a final guard, map an unexpected HOLD by a simple 0.5 threshold
        if rec2 == "HOLD":
            return "BUY" if xgb_prob >= 0.5 else "SELL"
        return rec2


# Compute a simple moving average with min_periods=1 (avoids early NaNs).
def _sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(int(n), min_periods=1).mean()


# From a close series, compute daily trend (UP/DOWN/FLAT) and momentum (fast vs slow SMA).
def _trend_and_momentum(close: pd.Series, fast: int = 10, slow: int = 30) -> Tuple[List[str], List[str]]:
    s = close.astype(float)
    fast_ma = _sma(s, fast)
    slow_ma = _sma(s, slow)
    # Momentum: relative position of fast and slow moving averages
    momentum = np.where(fast_ma > slow_ma, "FAST_ABOVE",
                 np.where(fast_ma < slow_ma, "FAST_BELOW", "UNDEFINED"))
    # Trend: compare price vs price N days ago
    lookback = slow
    shifted = s.shift(lookback - 1).fillna(method="bfill")
    trend = np.where(s > shifted, "UP", np.where(s < shifted, "DOWN", "FLAT"))
    return trend.tolist(), momentum.tolist()


# Compute max drawdown and its duration (days) from an equity curve.
def _max_drawdown(equity: pd.Series) -> Tuple[float, int]:
    if equity.empty:
        return 0.0, 0
    run_max = equity.cummax()
    dd = (equity - run_max) / run_max
    min_idx = dd.idxmin()                # date of deepest drawdown
    max_dd = float(dd.min()) if not dd.empty else 0.0
    # Duration: days between peak before that min and the min itself
    try:
        start_idx = (equity.loc[:min_idx] == run_max.loc[:min_idx]).loc[lambda s: s].index[-1]
        duration_days = int((min_idx - start_idx).days)
    except Exception:
        duration_days = 0
    return max_dd, duration_days


# Generate daily TRF signals using Transformer + XGBoost + CrewAI over [start, end].
def build_trf_signals(
    ticker: str,
    start: str,
    end: str,
    transformer_path: str = DEFAULT_TRANSFORMER_PATH,
    xgb_model_path: str = DEFAULT_XGB_PATH,
    window_size: int = 20,
    device: Optional[str] = None,
    buy_hi: float = 0.525,
    sell_lo: float = 0.475,
) -> pd.DataFrame:
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # Sanity-check model checkpoints exist
    if not os.path.isfile(transformer_path):
        raise FileNotFoundError(f"Missing Transformer weights: {transformer_path}")
    if not os.path.isfile(xgb_model_path):
        raise FileNotFoundError(f"Missing XGBoost model: {xgb_model_path}")

    # Reuse UI data utilities to fetch and normalize Close prices
    handler = DataHandler(ticker, start, end, window_size=window_size)
    close_df = handler.fetch_ohlcv()
    close_df = handler.impute_and_align(close_df)
    norm_df = handler.normalize(close_df)
    dates = norm_df.index

    # Ensure we have a Series named "close" for joins
    raw_close = close_df["Close"]
    if isinstance(raw_close, pd.DataFrame):
        raw_close = raw_close.iloc[:, 0]
    raw_close = raw_close.astype(float)
    if not isinstance(raw_close.index, pd.DatetimeIndex):
        raw_close.index = pd.to_datetime(raw_close.index)
    raw_close.name = "close"

    # Precompute daily context features for the agent
    trend_list, mom_list = _trend_and_momentum(raw_close)

    # Load trained models
    model = TransformerModel(feature_dim=1)
    model.load_state_dict(torch.load(transformer_path, map_location=device))
    model = model.to(device).eval()
    xgb = joblib.load(xgb_model_path)

    # CrewAI agent instance with configured thresholds
    agent = ForceActionCrewAgent(buy_hi=buy_hi, sell_lo=sell_lo)

    rows = []
    with torch.no_grad():
        # Walk forward day by day and compute windowed predictions
        for idx, dt in enumerate(dates):
            if idx < window_size - 1:
                # Not enough history to form a window
                rows.append({"date": dt, "recommendation": "HOLD",
                             "transformer_score": np.nan, "xgb_prob": np.nan})
                continue

            # Build the last `window_size` normalized values as input to the Transformer
            window = norm_df.iloc[idx - window_size + 1: idx + 1].values
            xb = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)

            # Transformer logits -> softmax probability for "up"
            logits = model(xb).cpu().numpy()
            prob_up = F.softmax(torch.tensor(logits), dim=1).numpy()[0, 1]

            # XGBoost meta probability over Transformer logits
            xgb_prob = float(xgb.predict_proba(logits)[0, 1])

            # Context for this date (trend/momentum)
            tr = trend_list[idx] if idx < len(trend_list) else "FLAT"
            mo = mom_list[idx] if idx < len(mom_list) else "UNDEFINED"

            # Query CrewAI; if it raises, pick a simple threshold fallback
            try:
                rec = agent.decide(ticker, xgb_prob, tr, mo)
            except Exception:
                rec = "BUY" if xgb_prob >= 0.5 else "SELL"

            rows.append({
                "date": dt,
                "recommendation": rec,
                "transformer_score": float(prob_up),
                "xgb_prob": float(xgb_prob)
            })

    sig = pd.DataFrame(rows).set_index("date").sort_index()

    # Final tidy-up: if any HOLD remains (e.g. warmup rows), fill via a 10/30 SMA overlay
    if (sig["recommendation"] == "HOLD").any():
        aligned = sig.join(raw_close, how="left").dropna(subset=["close"])
        fast = _sma(aligned["close"], 10)
        slow = _sma(aligned["close"], 30)
        overlay = np.where(fast > slow, "BUY", "SELL")
        mask = aligned["recommendation"] == "HOLD"
        aligned.loc[mask, "recommendation"] = overlay[mask]
        sig = aligned.drop(columns=["close"])

    return sig


# Simulate a daily long/short policy using next-day returns and simple commissions.
def simulate_long_short(
    signals: pd.DataFrame,
    close_series: pd.Series,
    initial_capital: float = 10_000.0,
    commission: float = 0.001,
) -> Tuple[pd.DataFrame, dict]:
    # Normalize price input to a 1-column Series with datetime index
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.iloc[:, 0]
    close_series = close_series.astype(float)
    if not isinstance(close_series.index, pd.DatetimeIndex):
        close_series.index = pd.to_datetime(close_series.index)

    # Align signals to prices and drop anything not BUY/SELL
    df = signals.join(close_series.rename("close"), how="inner").copy()
    df = df.sort_index()
    df = df[df["recommendation"].isin(["BUY", "SELL"])]

    # Nothing to simulate → flat equity
    if df.empty:
        equity = pd.Series([initial_capital], index=close_series.index[:1], name="equity")
        return equity.to_frame(), {
            "cumulative_return": 0.0, "sharpe": None,
            "max_drawdown": 0.0, "dd_duration": 0,
            "start_value": initial_capital, "end_value": initial_capital
        }

    # Use next-day simple returns to avoid look-ahead bias
    ret_next = close_series.pct_change().shift(-1).reindex(df.index)
    df["ret_next"] = ret_next
    df = df.dropna(subset=["ret_next"]).copy()
    if df.empty:
        equity = pd.Series([initial_capital], index=close_series.index[:1], name="equity")
        return equity.to_frame(), {
            "cumulative_return": 0.0, "sharpe": None,
            "max_drawdown": 0.0, "dd_duration": 0,
            "start_value": initial_capital, "end_value": initial_capital
        }

    # Walk forward: apply commission on position changes, then next-day PnL
    equity = float(initial_capital)
    pos = 0  # -1 short, +1 long
    eq_rows = []
    for dt, row in df.iterrows():
        target = 1 if row["recommendation"] == "BUY" else -1
        if target != pos:
            equity *= (1.0 - commission)  # one-way cost on turnover
            pos = target
        r_next = float(row["ret_next"])
        equity *= (1.0 + pos * r_next)
        eq_rows.append({"date": dt, "equity": equity})

    eq_df = pd.DataFrame(eq_rows).set_index("date").sort_index()

    # Basic stats: cumulative return, Sharpe (daily), and drawdown
    daily = eq_df["equity"].pct_change().fillna(0.0)
    cumret = float(eq_df["equity"].iloc[-1] / initial_capital - 1.0)
    vol = float(daily.std(ddof=0))
    sharpe = float((daily.mean() / vol) * np.sqrt(252)) if vol > 0 else None
    mdd, dd_dur = _max_drawdown(eq_df["equity"])

    return eq_df, {
        "cumulative_return": cumret,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "dd_duration": dd_dur,
        "start_value": float(initial_capital),
        "end_value": float(eq_df['equity'].iloc[-1]),
    }


# Orchestrate signal generation, simulation, and print a concise summary.
def run_backtest(
    ticker: str,
    start: str,
    end: str,
    transformer_path: str = DEFAULT_TRANSFORMER_PATH,
    xgb_model_path: str = DEFAULT_XGB_PATH,
    cash: float = 10_000.0,
    commission: float = 0.001,
    window_size: int = 20,
):
    # 1) Build TRF signals for the requested date range
    sig = build_trf_signals(
        ticker=ticker,
        start=start,
        end=end,
        transformer_path=transformer_path,
        xgb_model_path=xgb_model_path,
        window_size=window_size,
        buy_hi=0.525,
        sell_lo=0.475,
    )

    # 2) Fetch raw Close prices for PnL calculation
    px = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)["Close"]
    if isinstance(px, pd.DataFrame):
        px = px.iloc[:, 0]
    px = pd.Series(px).dropna()
    if not isinstance(px.index, pd.DatetimeIndex):
        px.index = pd.to_datetime(px.index)

    # 3) Simulate the policy and collect performance metrics
    equity_df, perf = simulate_long_short(
        signals=sig,
        close_series=px,
        initial_capital=cash,
        commission=commission,
    )

    # 4) Print compact results only (keeps console clean)
    print("===== TRF Backtest Summary =====")
    print(f"Sharpe Ratio: {perf['sharpe']:.2f}" if perf['sharpe'] is not None else "Sharpe Ratio: N/A")
    print(f"Total Return: {perf['cumulative_return'] * 100:.2f}%")
    daily_avg = equity_df["equity"].pct_change().mean() if not equity_df.empty else np.nan
    if np.isfinite(daily_avg):
        print(f"Average Daily Return: {daily_avg * 100:.2f}%")
        ann = ((1 + daily_avg) ** 252 - 1) * 100
        print(f"Implied Annual Return: {ann:.2f}%")
    else:
        print("Average Daily Return: N/A")
    print(f"Max Drawdown: {perf['max_drawdown'] * 100:.2f}%")
    print(f"Max Drawdown Duration: {perf['dd_duration']}")
    print(f"Final Portfolio Value: {perf['end_value']:.2f}")


# Minimal interactive entrypoint to run the backtest end-to-end.
if __name__ == "__main__":
    tic = input("Enter stock ticker (e.g. AAPL): ").strip().upper()
    s = input("Enter start date (YYYY-MM-DD): ").strip()
    e = input("Enter end date (YYYY-MM-DD): ").strip()

    run_backtest(
        ticker=tic,
        start=s,
        end=e,
        transformer_path=DEFAULT_TRANSFORMER_PATH,
        xgb_model_path=DEFAULT_XGB_PATH,
        cash=10_000.0,
        commission=0.001,
        window_size=20,
    )
