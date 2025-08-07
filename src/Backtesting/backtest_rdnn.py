# File: src/Backtesting/backtest_rdnn.py

import os
import sys
import logging
import json
import numpy as np
import pandas as pd
import torch
import backtrader as bt

# ensure project root is on path so we can import src.UI.rdnn for fetch_ohlcv and CrewAIDecisionAgent
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, HERE)

from src.UI.rdnn import fetch_ohlcv, CrewAIDecisionAgent
from sb3_contrib import RecurrentPPO

# -----------------------------------------------------------------------------
# Signal builder: run RL inference (and optional CrewAI override) to generate
# BUY/SELL/HOLD signals on a sliding window over OHLCV data.
# -----------------------------------------------------------------------------
def build_rdnn_signals(
    ticker: str,
    start: str,
    end: str,
    model_path: str,
    window_size: int = 10,
    use_crew_ai: bool = False
) -> pd.DataFrame:
    # Fetch raw OHLCV data and build date index
    raw = fetch_ohlcv(ticker, start, end, interval='1d')
    dates = pd.date_range(start, periods=raw.shape[0], freq='D')

    # Load trained recurrent PPO model once
    model = RecurrentPPO.load(model_path)
    hidden_state = None
    crew_agent = CrewAIDecisionAgent() if use_crew_ai else None

    records = []
    for i in range(window_size, len(raw)):
        window = raw[i - window_size : i]

        # 1) run model.predict to update hidden_state
        episode_starts = np.array([hidden_state is None], dtype=bool)
        _, hidden_state = model.predict(
            window[np.newaxis, ...],
            state=hidden_state,
            episode_start=episode_starts
        )

        # 2) convert hidden_state (np.ndarray) to torch tensors for distribution
        if isinstance(hidden_state, tuple):
            h_np, c_np = hidden_state
            h = torch.from_numpy(h_np).float()
            c = torch.from_numpy(c_np).float()
            hidden_state = (h, c)

        # 3) get action distribution
        obs_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
        ep_tensor = torch.tensor(episode_starts, dtype=torch.float32)
        dist, new_states = model.policy.get_distribution(
            obs_tensor,
            hidden_state,
            ep_tensor
        )
        hidden_state = new_states

        probs = dist.distribution.probs.detach().cpu().numpy()[0]

        # Map to recommendation
        idx = int(np.argmax(probs))
        rec_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
        recommendation = rec_map[idx]

        # Optional CrewAI override
        if crew_agent and crew_agent.llm_available:
            override = crew_agent.decide(ticker, json.dumps({"action_probs": probs.tolist()})) \
                                 .get("recommendation", "")
            if override in rec_map.values():
                recommendation = override

        records.append({
            "date":   dates[i].date(),
            "recommendation": recommendation,
            "prob_sell": float(probs[0]),
            "prob_hold": float(probs[1]),
            "prob_buy":  float(probs[2]),
        })

    sig_df = pd.DataFrame(records).set_index("date")
    return sig_df

# -----------------------------------------------------------------------------
# Backtrader strategy: executes signals from build_rdnn_signals
# -----------------------------------------------------------------------------
class RDNNSignalStrategy(bt.Strategy):
    params = dict(
        signal_df=None,    # DataFrame of signals
        allocation=0.95
    )

    def __init__(self):
        if self.p.signal_df is None:
            raise ValueError("signal_df must be provided")
        self.signal_map = {"BUY": 1, "SELL": -1, "HOLD": 0}
        self.order = None

    def next(self):
        dt = self.datas[0].datetime.date(0)
        rec = "HOLD"
        if dt in self.p.signal_df.index:
            rec = self.p.signal_df.loc[dt, "recommendation"]
        if self.order:
            return
        current = self.position.size
        target = self.signal_map.get(rec, 0)
        price = self.data.close[0]
        cash = self.broker.getcash()

        if target == 0:
            return
        # close opposite
        if target > 0 and current < 0:
            self.close()
        elif target < 0 and current > 0:
            self.close()
        # open if flat
        if current == 0 and target != 0:
            size = int((cash * self.p.allocation) // price)
            if size > 0:
                if target > 0:
                    self.order = self.buy(size=size)
                else:
                    self.order = self.sell(size=size)

# -----------------------------------------------------------------------------
# Runner: prompt user, run backtest, and print summary
# -----------------------------------------------------------------------------
def run_backtest(
    ticker: str,
    start: str,
    end: str,
    model_path: str,
    window_size: int = 10,
    use_crew_ai: bool = False,
    cash: float = 10000.0,
    commission: float = 0.001
):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger(__name__)

    logger.info(f"Building RDNN signals for {ticker} from {start} to {end}")
    sig_df = build_rdnn_signals(
        ticker=ticker,
        start=start,
        end=end,
        model_path=model_path,
        window_size=window_size,
        use_crew_ai=use_crew_ai
    )

    # Prepare price feed
    raw = fetch_ohlcv(ticker, start, end, interval='1d')
    price_df = pd.DataFrame(
        raw,
        columns=["Open", "High", "Low", "Close", "Volume"],
        index=pd.date_range(start, periods=raw.shape[0], freq='D')
    )
    price_df.index = pd.to_datetime(price_df.index)

    data_feed = bt.feeds.PandasData(dataname=price_df)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(RDNNSignalStrategy, signal_df=sig_df)
    cerebro.adddata(data_feed)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.01)
    cerebro.addanalyzer(bt.analyzers.Returns,     _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown,    _name='drawdown')

    logger.info("Running backtest...")
    results = cerebro.run()
    strat = results[0]

    sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', None)
    total = strat.analyzers.returns.get_analysis().get('rtot', None)
    avg   = strat.analyzers.returns.get_analysis().get('ravg', None)
    dd    = strat.analyzers.drawdown.get_analysis().get('maxdrawdown', None)
    dd_dur= strat.analyzers.drawdown.get_analysis().get('maxdrawdownperiod', None)
    final = cerebro.broker.getvalue()

    print("\n===== RDNN Backtest Summary =====")
    print(f"Sharpe Ratio:          {sharpe:.2f}" if sharpe is not None else "Sharpe Ratio: N/A")
    print(f"Total Return:          {total*100:.2f}%" if total is not None else "Total Return: N/A")
    if avg is not None:
        print(f"Avg Daily Return:      {avg*100:.2f}%")
        print(f"Implied Annual Return: {((1+avg)**252-1)*100:.2f}%")
    else:
        print("Avg Daily Return:      N/A")
    print(f"Max Drawdown:          {dd*100:.2f}%" if dd is not None else "Max Drawdown: N/A")
    print(f"Max Drawdown Duration: {dd_dur}")
    print(f"Final Portfolio Value: {final:.2f}")

    try:
        cerebro.plot()
    except Exception:
        pass

if __name__ == "__main__":
    ticker     = input("Enter ticker symbol (e.g., AAPL): ").strip().upper()
    start      = input("Enter start date (YYYY-MM-DD): ").strip()
    end        = input("Enter end date (YYYY-MM-DD): ").strip()
    model_path = input("Enter path to trained recurrent PPO model zip: ").strip()
    run_backtest(
        ticker=ticker,
        start=start,
        end=end,
        model_path=model_path
    )
