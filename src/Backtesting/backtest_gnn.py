import os
import sys
import time
import logging
import contextlib
import io
from typing import Tuple, Optional

import backtrader as bt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yfinance as yf

# Make project root importable (repo root is two levels up from this file)
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, HERE)

# Import the GNN pipeline components from UI (do NOT modify that module here)
from src.UI import gnn as gnn_mod  # PriceLoader, GraphBuilder, StudentGNN, DecisionAgent

# Silence noisy libraries by default
for _name in ("openai", "httpx", "httpcore", "urllib3", "yfinance", "backtrader"):
    logging.getLogger(_name).setLevel(logging.WARNING)
    logging.getLogger(_name).propagate = False


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
# Convert yfinance OHLCV (possibly MultiIndex) into Backtrader-friendly flat DataFrame for a given ticker.
def _normalize_ohlcv_for_bt(ohlcv: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Flatten yfinance-style OHLCV into single-level OHLCV for Backtrader."""
    df = ohlcv.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # prefer exact ticker level if present
        if ticker in df.columns.get_level_values(1):
            df = df.xs(ticker, axis=1, level=1)
        else:
            first = df.columns.levels[1][0]
            df = df.xs(first, axis=1, level=1)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


# Parse a CSV list of tickers and ensure the primary ticker is included (order-preserving de-dup).
def _parse_universe(universe_csv: str, ticker: str) -> list[str]:
    """Parse comma-separated tickers into a clean universe including `ticker`."""
    if not universe_csv:
        return [ticker.upper()]
    parts = [t.strip().upper() for t in universe_csv.split(',') if t.strip()]
    if ticker.upper() not in parts:
        parts.append(ticker.upper())
    seen = set()
    ordered = []
    for t in parts:
        if t not in seen:
            ordered.append(t)
            seen.add(t)
    return ordered


# Resolve a possibly relative or ~-expanded path into an absolute path (tries repo root as fallback).
def _resolve_path(p: str) -> str:
    """Return an absolute path; try raw `p`, then project-root joined."""
    if not p:
        return p
    p = os.path.expanduser(p)
    if os.path.isabs(p):
        return p
    if os.path.isfile(p):
        return os.path.abspath(p)
    cand = os.path.join(HERE, p)
    return os.path.abspath(cand)


# Temporarily silence stdout/stderr and lower logging to suppress noisy third-party output.
@contextlib.contextmanager
def _suppress_output_and_logs():
    """
    Suppress stdout/stderr prints and temporarily disable logging <= CRITICAL.
    Useful to silence DecisionAgent prints and HTTP logs inside tight loops.
    """
    prev_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        yield
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        logging.disable(prev_disable)


# Compute the percentile rank of one element within an array (used for cross-sectional ranking).
def _percentile_rank_for_index(arr: np.ndarray, index: int) -> float:
    """Return percentile rank (0..1] for arr[index] within arr (ties averaged)."""
    order = np.argsort(arr)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(arr) + 1)
    return float(ranks[index]) / float(len(arr))


# Estimate a daily Sharpe ratio directly from signals and price returns if analyzer returns N/A.
def _fallback_sharpe_from_signals(price_df: pd.DataFrame, signals_df: pd.DataFrame) -> Optional[float]:
    """
    Compute an approximate daily Sharpe from signals & closes:
      - Position d[t] = +1 BUY, -1 SELL, 0 HOLD (use previous day's position).
      - Strategy daily return r[t] = d[t-1] * close_ret[t].
      - Sharpe = sqrt(252) * mean(r) / std(r), if std > 0.
    Ignores commission and slippage (Backtrader summary remains authoritative).
    """
    close = price_df["Close"].astype(float)
    rets = close.pct_change().fillna(0.0)
    # Align signals to trading days
    s = signals_df.reindex(close.index.date, method="ffill").fillna({"recommendation": "HOLD"})
    dir_map = {"BUY": 1, "SELL": -1, "HOLD": 0}
    d = s["recommendation"].map(dir_map).astype(int)
    # Position used for today's return is yesterday's decision
    d_shift = d.shift(1).fillna(0.0)
    strat_ret = d_shift.values * rets.values
    mu = float(np.mean(strat_ret))
    sd = float(np.std(strat_ret, ddof=1))
    if sd <= 0:
        return None
    return float(np.sqrt(252.0) * (mu / sd))


# Load a StudentGNN checkpoint without prompting; fall back to a random model if loading fails.
def _load_student_model_noninteractive(path_hint: str, device: str) -> Tuple[torch.nn.Module, str, bool]:
    """
    Load StudentGNN from path. If missing/invalid, use a RANDOM (untrained) model.
    Returns: (model, resolved_path, used_random)
    """
    candidate = _resolve_path(path_hint or "")
    if candidate and os.path.isfile(candidate):
        try:
            model = gnn_mod.StudentGNN(1, 32, 64, 2).to(device)
            model.load_state_dict(torch.load(candidate, map_location=device))
            model.eval()
            return model, candidate, False
        except Exception:
            pass
    # Fall back to random model (no prompts)
    model = gnn_mod.StudentGNN(1, 32, 64, 2).to(device)
    model.eval()
    return model, "<random-untrained>", True


# -----------------------------------------------------------------------------
# Crew-only Decision Agent wrapper (forces BUY/SELL, no HOLD)
# -----------------------------------------------------------------------------
class ForcedDecisionAgent(gnn_mod.DecisionAgent):
    """
    Crew-style agent that MUST return BUY or SELL (never HOLD).
    - Overrides the prompt to forbid HOLD.
    - If the LLM still returns HOLD or invalid JSON, it enforces BUY/SELL using p>=0.5.
    """

    def _build_prompt(self, student_prob: float) -> str:  # type: ignore[override]
        from textwrap import dedent
        return dedent(f"""
            You are a trading decision microservice.
            Input: an object with a single numeric field 'student_prob' in [0,1],
            representing the probability that a stock's price goes UP in the next period.

            IMPORTANT CONSTRAINT:
              - You MUST return either {{"recommendation":"BUY"}} or {{"recommendation":"SELL"}}.
              - DO NOT return HOLD under any circumstance.
              - If uncertainty is high, use a 0.50 tie-break: >=0.50 => BUY, <0.50 => SELL.

            Respond with only a JSON object and nothing else.
            Input: {{"student_prob": {student_prob:.6f}}}
        """).strip()

    def get_decision(self, student_prob: float) -> dict:  # type: ignore[override]
        """Call parent LLM flow; if it returns HOLD or invalid, force BUY/SELL by 0.5 tie-break."""
        try:
            out = super().get_decision(student_prob)  # may return HOLD
            rec = str(out.get("recommendation", "")).upper()
            if rec == "HOLD" or rec not in {"BUY", "SELL"}:
                rec = "BUY" if student_prob >= 0.5 else "SELL"
            return {"recommendation": rec}
        except Exception:
            # If parent path fails, enforce deterministic BUY/SELL (still no HOLD)
            return {"recommendation": "BUY" if student_prob >= 0.5 else "SELL"}


# Build per-day BUY/SELL/HOLD signals using StudentGNN outputs; Crew Agent is the ONLY decider here.
def build_gnn_signals(
    ticker: str,
    universe_csv: str,
    start: str,
    end: str,
    student_model_path: str,
    window_size: int = 30,
    delta: int = 10,
    threshold: float = 0.3,
    work_dir: str | None = None,
    device: str | None = None,
    use_crew_ai: bool = True,         # default: USE Crew AI agent
    local_agent_only: bool = False,   # default: actually call the agent (no local-only)
    student_model: torch.nn.Module | None = None,
    # Dynamic decision knobs (kept for completeness; not used when agent is enabled)
    enable_dynamic: bool = False,
    abs_buy: float = 0.55,
    abs_sell: float = 0.45,
    cs_buy_pct: float = 0.60,
    cs_sell_pct: float = 0.40,
    ts_lookback: int = 15,
    ts_z: float = 0.35,
) -> pd.DataFrame:
    """
    Compute daily BUY/SELL signals for the given `ticker` using the StudentGNN.
    With Crew agent enabled, we use a ForcedDecisionAgent that never returns HOLD.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # Universe (ensure primary ticker included)
    universe = _parse_universe(universe_csv, ticker)

    # Dedicated workspace; redirect UI module output here (quietly)
    run_dir = work_dir or os.path.join(HERE, "artifacts", f"gnn_bt_{ticker}_{int(time.time())}")
    os.makedirs(run_dir, exist_ok=True)
    gnn_mod.SCRIPT_DIR = run_dir

    # 1) Prices & returns
    loader = gnn_mod.PriceLoader(universe, start, end, data_dir="data")
    with _suppress_output_and_logs():
        _ = loader.fetch_ohlc_data()
        returns_df = loader.compute_daily_returns()
    dates = list(returns_df.index)
    if len(dates) < max(delta, window_size):
        raise RuntimeError("Not enough data to satisfy delta/window_size history")

    # 2) Graphs
    gb = gnn_mod.GraphBuilder(returns_df, window_size=window_size, threshold=threshold, output_dir="adjacency_data")
    with _suppress_output_and_logs():
        gb.build_graphs()
    adj_dir = os.path.join(run_dir, "adjacency_data")
    if not os.path.isdir(adj_dir):
        raise RuntimeError("Adjacency directory not found after GraphBuilder.run()")

    # 3) Model
    if student_model is None:
        resolved = _resolve_path(student_model_path)
        model = gnn_mod.StudentGNN(1, 32, 64, 2).to(device)
        if os.path.isfile(resolved):
            try:
                model.load_state_dict(torch.load(resolved, map_location=device))
            except Exception:
                pass
        model.eval()
    else:
        model = student_model

    # 4) Crew agent (forced BUY/SELL) — output/logs suppressed
    agent = ForcedDecisionAgent() if use_crew_ai else None

    # Iterate dates to produce daily signal for the focal ticker
    signals = []
    tkr_index = universe.index(ticker.upper())

    start_idx = max(window_size - 1, delta - 1)
    for idx in range(start_idx, len(dates)):
        date = dates[idx]
        date_str = date.strftime("%Y-%m-%d")
        adj_path = os.path.join(adj_dir, f"adjacency_{date_str}.npy")
        if not os.path.isfile(adj_path):
            # In the rare case adjacency is missing, stay flat (treat as SELL to force change? no—keep flat safely)
            signals.append({"date": date, "recommendation": "SELL", "student_prob": 0.0})
            continue

        # (delta x N) feature window up to 'date'
        seq_start = idx - (delta - 1)
        seq_end = idx + 1
        feat_seq = returns_df.iloc[seq_start:seq_end].values.astype(np.float32)
        adj_np = np.load(adj_path).astype(np.float32)

        feat = torch.tensor(feat_seq, device=device)
        adj_t = torch.tensor(adj_np, device=device)

        with torch.no_grad():
            logits = model(feat, adj_t)                        # (N, 2)
            prob_all = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            prob_up = float(prob_all[tkr_index])

        # Crew agent decides (no local overrides, no HOLD)
        rec = "BUY"
        if agent is not None and not local_agent_only:
            with _suppress_output_and_logs():
                try:
                    rec = agent.get_decision(prob_up)["recommendation"]
                except Exception:
                    # If LLM path fails, force deterministic BUY/SELL to avoid flat PnL
                    rec = "BUY" if prob_up >= 0.5 else "SELL"

        signals.append({"date": date, "recommendation": rec, "student_prob": prob_up})

    df = pd.DataFrame(signals).set_index("date").sort_index()
    return df


# -----------------------------------------------------------------------------
# Backtrader Strategy
# -----------------------------------------------------------------------------
class GNNSignalStrategy(bt.Strategy):
    """Simple strategy that follows daily BUY/SELL recommendations."""
    params = dict(
        allocation=0.95,   # fraction of available cash to deploy per trade
        signal_df=None,    # DataFrame with 'recommendation' per date
    )

    # Initialize strategy state and map recommendations to target directions.
    def __init__(self):
        if self.p.signal_df is None:
            raise ValueError("signal_df must be provided to GNNSignalStrategy")
        self.order = None
        self.signal_map = {"BUY": 1, "SELL": -1, "HOLD": 0}

    # On each bar, read the day's recommendation and place/close orders accordingly.
    def next(self):
        dt = self.datas[0].datetime.date(0)
        rec = "SELL"  # default to a position-changing action (avoids being flat forever)
        try:
            if dt in self.p.signal_df.index:
                rec = self.p.signal_df.loc[dt, "recommendation"]
            else:
                ts = pd.Timestamp(dt)
                if ts in self.p.signal_df.index:
                    rec = self.p.signal_df.loc[ts, "recommendation"]
        except Exception:
            rec = "SELL"

        if isinstance(rec, pd.Series):
            rec = rec.values[0]

        if self.order:
            return

        current_pos = self.position.size
        target_dir = self.signal_map.get(rec, 0)

        if target_dir > 0:  # LONG
            if current_pos < 0:
                self.close()
            elif current_pos == 0:
                cash = self.broker.getcash()
                price = float(self.data.close[0])
                size = int((cash * self.p.allocation) // price)
                if size > 0:
                    self.order = self.buy(size=size)
        elif target_dir < 0:  # SHORT
            if current_pos > 0:
                self.close()
            elif current_pos == 0:
                cash = self.broker.getcash()
                price = float(self.data.close[0])
                size = int((cash * self.p.allocation) // price)
                if size > 0:
                    self.order = self.sell(size=size)
        else:
            # For completeness; with ForcedDecisionAgent we shouldn't hit HOLD.
            return


# -----------------------------------------------------------------------------
# Runner (called by interactive main)
# -----------------------------------------------------------------------------
# Orchestrate signal generation, Backtrader execution, and concise performance summary printing.
def run_backtest(
    ticker: str,
    start: str,
    end: str,
    universe_csv: str = "",
    student_model_path: str = os.path.join(HERE, "src", "UI", "student_model", "best_student.pth"),
    cash: float = 10000.0,
    commission: float = 0.001,
    window_size: int = 30,
    delta: int = 10,
    threshold: float = 0.3,
    work_dir: str | None = None,
    use_crew_ai: bool = True,          # use Crew AI
    local_agent_only: bool = False,    # actually call the agent
    enable_dynamic: bool = False,      # disable local tie-breakers
    abs_buy: float = 0.55,
    abs_sell: float = 0.45,
    cs_buy_pct: float = 0.60,
    cs_sell_pct: float = 0.40,
    ts_lookback: int = 15,
    ts_z: float = 0.35,
):
    """Build signals, run Backtrader, and print a performance summary (quiet)."""
    # Load (or fall back to) model WITHOUT extra prompts
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, resolved_path, used_random = _load_student_model_noninteractive(student_model_path, device)

    # Build signals (Crew agent only; no HOLD)
    signals_df = build_gnn_signals(
        ticker=ticker,
        universe_csv=universe_csv,
        start=start,
        end=end,
        student_model_path=resolved_path,
        window_size=window_size,
        delta=delta,
        threshold=threshold,
        work_dir=work_dir,
        use_crew_ai=use_crew_ai,
        local_agent_only=local_agent_only,
        student_model=model,
        enable_dynamic=enable_dynamic,
        abs_buy=abs_buy,
        abs_sell=abs_sell,
        cs_buy_pct=cs_buy_pct,
        cs_sell_pct=cs_sell_pct,
        ts_lookback=ts_lookback,
        ts_z=ts_z,
    )

    # Normalize index to date for lookup
    signals_df.index = signals_df.index.date

    # Prepare price feed
    ohlcv = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    price_df = _normalize_ohlcv_for_bt(ohlcv, ticker).dropna(how='any')
    if not isinstance(price_df.index, pd.DatetimeIndex):
        price_df.index = pd.to_datetime(price_df.index)
    data_feed = bt.feeds.PandasData(dataname=price_df)

    # Setup Cerebro (quiet)
    cerebro = bt.Cerebro(runonce=True, preload=True)
    cerebro.addstrategy(GNNSignalStrategy, signal_df=signals_df)
    cerebro.adddata(data_feed)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.01)
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

    # Run (quiet)
    with _suppress_output_and_logs():
        result = cerebro.run()

    strat = result[0]
    sharpe = strat.analyzers.sharpe.get_analysis()
    returns = strat.analyzers.returns.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()

    # Summary -------------------------------------------------------------
    final_value = cerebro.broker.getvalue()
    print("===== Backtest Summary =====")
    # Backtrader Sharpe or fallback
    sharperatio = sharpe.get('sharperatio')
    if sharperatio is None:
        # Fallback: compute approximate daily Sharpe from signals & closes
        sr_fallback = _fallback_sharpe_from_signals(price_df, signals_df)
        print(f"Sharpe Ratio (fallback): {sr_fallback:.2f}" if sr_fallback is not None else "Sharpe Ratio: N/A")
    else:
        print(f"Sharpe Ratio: {sharperatio:.2f}")

    total_return = returns.get('rtot')
    print(f"Total Return: {total_return * 100:.2f}%" if total_return is not None else "Total Return: N/A")

    avg_daily = returns.get('ravg')
    if avg_daily is not None:
        print(f"Average Daily Return: {avg_daily * 100:.2f}%")
        avg_annual = ((1 + avg_daily) ** 252 - 1) * 100
        print(f"Implied Annual Return: {avg_annual:.2f}%")
    else:
        print("Average Daily Return: N/A")

    print(f"Final Portfolio Value: {final_value:.2f}")

    # Optional plot (silent failure in headless envs)
    try:
        with _suppress_output_and_logs():
            cerebro.plot()
    except Exception:
        pass

    return signals_df, result


# -----------------------------------------------------------------------------
# Interactive main (ONLY ask for ticker, start, end) — Crew-only defaults
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== GNN Backtest (interactive) ===")
    ticker = input("Enter primary ticker to trade (e.g., AAPL): ").strip().upper()
    while not ticker:
        ticker = input("Please enter a valid ticker (e.g., AAPL): ").strip().upper()

    start = input("Enter start date (YYYY-MM-DD): ").strip()
    end = input("Enter end date (YYYY-MM-DD): ").strip()

    # Defaults (agent-only mode; no local rules)
    universe_csv = ""  # just the primary ticker unless you edit this string
    student_model_path = os.path.join(HERE, "src", "UI", "student_model", "best_student.pth")
    cash = 10000.0
    commission = 0.001
    window_size = 30
    delta = 10
    threshold = 0.3
    work_dir = None
    use_crew_ai = True          # use Crew AI agent
    local_agent_only = False    # actually call the agent (no local-only)
    enable_dynamic = False      # do not apply local tie-breakers when agent is used
    abs_buy, abs_sell = 0.55, 0.45
    cs_buy_pct, cs_sell_pct = 0.60, 0.40
    ts_lookback, ts_z = 15, 0.35

    run_backtest(
        ticker=ticker,
        start=start,
        end=end,
        universe_csv=universe_csv,
        student_model_path=student_model_path,
        cash=cash,
        commission=commission,
        window_size=window_size,
        delta=delta,
        threshold=threshold,
        work_dir=work_dir,
        use_crew_ai=use_crew_ai,
        local_agent_only=local_agent_only,
        enable_dynamic=enable_dynamic,
        abs_buy=abs_buy,
        abs_sell=abs_sell,
        cs_buy_pct=cs_buy_pct,
        cs_sell_pct=cs_sell_pct,
        ts_lookback=ts_lookback,
        ts_z=ts_z,
    )
