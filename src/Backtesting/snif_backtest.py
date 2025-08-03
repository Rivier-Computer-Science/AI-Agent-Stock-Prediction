# File: src/Backtesting/snif_backtest.py
import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta

import backtrader as bt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Ensure project root is on path so imports from src.UI.snif work
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, HERE)

# SNIF components (must be importable from your codebase)
from src.UI.snif import ReturnFetcher
from src.UI.snif import AutoencoderTrainer
from src.UI.snif import TopologyBuilder
from src.UI.snif import GCN_LSTM
from src.UI.snif import CrewAIDecisionAgent

# Module-level logger
logger = logging.getLogger(__name__)


####################################
# Helpers
####################################
def _normalize_ohlcv_for_bt(ohlcv: pd.DataFrame, ticker):
    """
    Flatten yfinance-style OHLCV output (which may be MultiIndex for multiple tickers)
    into a single-level DataFrame with columns like Open/High/Low/Close/Volume that
    Backtrader expects.
    """
    df = ohlcv.copy()
    if isinstance(df.columns, pd.MultiIndex):
        key = ticker if isinstance(ticker, str) else (ticker[0] if ticker else df.columns.levels[0][0])
        if key in df.columns.get_level_values(0):
            df = df[key]
        else:
            first = df.columns.levels[0][0]
            df = df[first]
    df = df.sort_index()
    return df


def _parse_label_order(label_str: str):
    """
    Parse comma-separated class label order from CLI into list of uppercase labels.
    Example: "SELL,HOLD,BUY" -> ["SELL","HOLD","BUY"]
    """
    parts = [p.strip().upper() for p in label_str.split(',')]
    if len(parts) < 1:
        raise ValueError("model-labels must be comma separated, e.g., SELL,HOLD,BUY")
    return parts


####################################
# Core: build SNIF signals with simple thresholds and optional CrewAI override
####################################
def build_snif_signals(
    ticker: str,
    start: str,
    end: str,
    encoder_path: str,
    scripted_model_path: str,
    label_order: list[str],
    window_size: int = 30,
    device: str = None,
    threshold: float = 0.7,
    use_crew_ai: bool = True,
    buy_threshold: float = 0.34,
    sell_threshold: float = 0.34,
):
    """
    Compute daily BUY/SELL/HOLD signals for the given ticker and date range.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.isfile(encoder_path):
        raise FileNotFoundError(f"Encoder file not found at {encoder_path}")
    if not os.path.isfile(scripted_model_path):
        raise FileNotFoundError(f"Scripted model not found at {scripted_model_path}")

    # Data ingestion and preprocessing
    fetcher = ReturnFetcher()
    ohlcv = fetcher.fetch_ohlcv(ticker, start, end)
    returns = fetcher.compute_returns(ohlcv)
    cleaned = fetcher.clean_and_align(returns)
    dates = cleaned.index

    # Prepare encoder and load trained weights
    input_dim = cleaned.shape[1]
    latent_dim = 16
    ae_trainer = AutoencoderTrainer(input_dim=input_dim, latent_dim=latent_dim)
    ae_trainer.device = device
    ae_trainer.model.to(device)
    ae_trainer.load_encoder(encoder_path)

    # Load scripted GCN+LSTM student model
    gcn_lstm = torch.jit.load(scripted_model_path, map_location=device)
    gcn_lstm.eval()

    # Topology builder
    topo = TopologyBuilder(threshold=threshold)

    # CrewAI agent (optional)
    crew_agent = None
    if use_crew_ai:
        try:
            crew_agent = CrewAIDecisionAgent()
        except Exception:
            crew_agent = None

    signals = []
    with torch.no_grad():
        full_tensor = torch.tensor(cleaned.values, dtype=torch.float32)
        emb_full = ae_trainer.extract_embeddings(full_tensor).numpy()

    # Embedding variance diagnostic (logged at debug level)
    emb_var = np.var(emb_full, axis=0)
    logger.debug("Embedding dims variance (first 10): %s", emb_var[:10].tolist())
    if np.all(emb_var < 1e-6):
        logger.warning("Embeddings have very low variance; encoder may be untrained or degenerate.")

    # Sliding window signal generation
    for idx in range(len(dates)):
        if idx < window_size - 1:
            rec = "HOLD"
            probs = {lbl: 0.0 for lbl in label_order}
            if "HOLD" in probs:
                probs["HOLD"] = 1.0
        else:
            window_emb = emb_full[idx - window_size + 1: idx + 1]
            sim = topo.compute_similarity(window_emb)
            adj_mat = topo.sparsify(sim)
            seq = torch.tensor(window_emb, dtype=torch.float32).unsqueeze(0).to(device)
            adj = torch.tensor(adj_mat, dtype=torch.float32).to(device)
            with torch.no_grad():
                logits = gcn_lstm(seq, adj).squeeze(0)
                probs_tensor = F.softmax(logits, dim=-1).cpu().numpy()

            # Map probabilities to labels
            probs = {}
            for i, lbl in enumerate(label_order):
                probs[lbl] = float(probs_tensor[i]) if i < len(probs_tensor) else 0.0

            # Heuristic signal based on thresholds
            if probs.get("BUY", 0.0) > buy_threshold:
                model_signal = "BUY"
            elif probs.get("SELL", 0.0) > sell_threshold:
                model_signal = "SELL"
            else:
                model_signal = "HOLD"

            rec = model_signal

            # CrewAI override logic: non-HOLD overrides model_signal
            if crew_agent:
                try:
                    representative_prob = probs.get("BUY", probs.get(model_signal, 0.0))
                    decision = crew_agent.decide([{"stock": ticker, "snif_prob": representative_prob}])
                    if isinstance(decision, list) and len(decision) > 0:
                        cand = decision[0].get("recommendation", "HOLD").upper()
                        if cand in label_order and cand != "HOLD":
                            rec = cand
                except Exception:
                    pass

        signals.append({
            "date": dates[idx],
            "recommendation": rec,
            "snif_probs": probs
        })

    df = pd.DataFrame(signals).set_index("date")
    probs_df = pd.DataFrame(df["snif_probs"].tolist(), index=df.index)
    df = df.drop(columns=["snif_probs"]).join(probs_df)

    # Warn if everything is HOLD (possible model degeneracy)
    if df["recommendation"].nunique() == 1 and df["recommendation"].unique()[0] == "HOLD":
        logger.warning("All recommendations are HOLD. Check model outputs or lower thresholds.")

    return df  # Contains recommendation and per-label probabilities


####################################
# Backtrader Strategy
####################################
class SNIFSignalStrategy(bt.Strategy):
    params = dict(
        allocation=0.95,   # fraction of available cash to deploy per trade
        signal_df=None,    # DataFrame of signals
    )

    def __init__(self):
        if self.p.signal_df is None:
            raise ValueError("signal_df must be provided to SNIFSignalStrategy")
        self.order = None
        self.signal_map = {"BUY": 1, "SELL": -1, "HOLD": 0}

    def next(self):
        dt = self.datas[0].datetime.date(0)
        rec = "HOLD"
        try:
            if dt in self.p.signal_df.index:
                rec = self.p.signal_df.loc[dt, "recommendation"]
            else:
                ts = pd.Timestamp(dt)
                if ts in self.p.signal_df.index:
                    rec = self.p.signal_df.loc[ts, "recommendation"]
        except Exception:
            rec = "HOLD"

        if isinstance(rec, pd.Series):
            rec = rec.values[0]

        if self.order:
            return

        current_pos = self.position.size
        target_dir = self.signal_map.get(rec, 0)

        if rec == "HOLD":
            return

        if target_dir > 0:  # LONG
            if current_pos < 0:
                self.close()
            elif current_pos == 0:
                cash = self.broker.getcash()
                price = self.data.close[0]
                size = int((cash * self.p.allocation) // price)
                if size > 0:
                    self.order = self.buy(size=size)
        elif target_dir < 0:  # SHORT
            if current_pos > 0:
                self.close()
            elif current_pos == 0:
                cash = self.broker.getcash()
                price = self.data.close[0]
                size = int((cash * self.p.allocation) // price)
                if size > 0:
                    self.order = self.sell(size=size)


####################################
# Runner
####################################
def run_backtest(
    ticker: str,
    start: str,
    end: str,
    encoder_path: str,
    scripted_model_path: str,
    label_order: list[str],
    cash: float = 10000,
    commission: float = 0.001,
    window_size: int = 30,
    buy_threshold: float = 0.34,
    sell_threshold: float = 0.34,
):
    """
    Generate SNIF signals and execute the backtest in Backtrader, then print performance summary.
    """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger.setLevel(logging.INFO)

    print(f"Building SNIF signals for {ticker} from {start} to {end} with window {window_size}")
    signal_df = build_snif_signals(
        ticker=ticker,
        start=start,
        end=end,
        encoder_path=encoder_path,
        scripted_model_path=scripted_model_path,
        label_order=label_order,
        window_size=window_size,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )

    # Normalize the index for strategy lookup
    signal_df.index = signal_df.index.date

    # Prepare price feed
    fetcher = ReturnFetcher()
    ohlcv = fetcher.fetch_ohlcv(ticker, start, end)
    price_df = _normalize_ohlcv_for_bt(ohlcv, ticker).dropna(how='any')
    if not isinstance(price_df.index, pd.DatetimeIndex):
        price_df.index = pd.to_datetime(price_df.index)

    data_feed = bt.feeds.PandasData(dataname=price_df)

    # Setup Cerebro
    cerebro = bt.Cerebro(runonce=True, preload=True)
    cerebro.addstrategy(SNIFSignalStrategy, signal_df=signal_df)
    cerebro.adddata(data_feed)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.01)
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

    logging.info(f"Running backtest for {ticker}...")
    result = cerebro.run()
    strat = result[0]
    sharpe = strat.analyzers.sharpe.get_analysis()
    returns = strat.analyzers.returns.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()

    # Summary printing
    print("===== Backtest Summary =====")
    sharperatio = sharpe.get('sharperatio')
    if sharperatio is not None:
        print(f"Sharpe Ratio: {sharperatio:.2f}")
    else:
        print("Sharpe Ratio: N/A")

    total_return = returns.get('rtot')
    if total_return is not None:
        print(f"Total Return: {total_return * 100:.2f}%")
    else:
        print("Total Return: N/A")

    avg_daily = returns.get('ravg')
    if avg_daily is not None:
        print(f"Average Daily Return: {avg_daily * 100:.2f}%")
        avg_annual = ((1 + avg_daily) ** 252 - 1) * 100
        print(f"Implied Annual Return: {avg_annual:.2f}%")
    else:
        print("Average Daily Return: N/A")

    md = drawdown.get('maxdrawdown')
    if md is not None:
        print(f"Max Drawdown: {md * 100:.2f}%")
    else:
        print("Max Drawdown: N/A")
    print(f"Max Drawdown Duration: {drawdown.get('maxdrawdownperiod', 'N/A')}")

    final_value = cerebro.broker.getvalue()
    print(f"Final Portfolio Value: {final_value:.2f}")

    # Plot if possible
    try:
        cerebro.plot()
    except Exception:
        pass


# CLI entrypoint
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SNIF backtest using CrewAI/SNIF signals")
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g., AAPL")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--encoder", default="checkpoints/autoencoder/best_encoder.pth", help="Path to encoder .pth")
    parser.add_argument("--scripted", default=os.path.join("models", "snif_student.pt"), help="Path to scripted GCN+LSTM .pt")
    parser.add_argument("--model-labels", default="SELL,HOLD,BUY", help="Comma-separated output order of model logits")
    parser.add_argument("--cash", type=float, default=10000, help="Initial cash")
    parser.add_argument("--commission", type=float, default=0.001, help="Per-trade commission")
    parser.add_argument("--window", type=int, default=30, help="Sliding window size for sequence")
    parser.add_argument("--buy-threshold", type=float, default=0.34, help="Threshold for BUY signal from model probability")
    parser.add_argument("--sell-threshold", type=float, default=0.34, help="Threshold for SELL signal from model probability")
    args = parser.parse_args()

    label_order = _parse_label_order(args.model_labels)
    run_backtest(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        encoder_path=args.encoder,
        scripted_model_path=args.scripted,
        label_order=label_order,
        cash=args.cash,
        commission=args.commission,
        window_size=args.window,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
    )
