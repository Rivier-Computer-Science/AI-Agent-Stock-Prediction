import numpy as np
import os
import pandas as pd
import backtrader as bt
import logging
from datetime import datetime
from src.Data_Retrieval.data_fetcher import DataFetcher
from src.Indicators.griffiths_predictor import GriffithsPredictor
from src.Indicators.cycle_trend_detection import CycleDetector

# Logging configuration
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global defaults for the Griffiths Predictor
GRIFFITHS_DEFAULTS = {
    'make_stationary': False,
    'use_log_diff': False,
    'length': 18,
    'lower_bound': 18,
    'upper_bound': 40,
    'bars_fwd': 2,
    'peak_decay': 0.991,
    'initial_peak': 0.0001,
    'scale_to_price': False,
    'allocation': 1.0
}

def dict_to_params(d: dict) -> tuple:
    """
    Convert a dict into a Backtrader 'params' tuple,
    e.g. { 'length': 18 } -> (('length', 18), ...)
    """
    return tuple((k, v) for k, v in d.items())

# -------------------------
# Custom Data Feed
# -------------------------
class MyPandasData(bt.feeds.PandasData):
    lines = ('is_trending',)
    params = (
        ('datetime', None),
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
        ('openinterest', None),
        ('is_trending', 'is_trending'),
    )

# -------------------------
# Griffiths Predictor Indicator
# -------------------------
class GriffithsPredictorBT(bt.Indicator):
    lines = ('gp_signal',)
    params = dict_to_params(GRIFFITHS_DEFAULTS)

    def __init__(self):
        self.addminperiod(self.p.upper_bound)
        close_prices = np.array(self.data.close)
        gp = GriffithsPredictor(
            close_prices=close_prices,
            make_stationary=self.p.make_stationary,
            use_log_diff=self.p.use_log_diff,
            length=self.p.length,
            lower_bound=self.p.lower_bound,
            upper_bound=self.p.upper_bound,
            bars_fwd=self.p.bars_fwd,
            peak_decay=self.p.peak_decay,
            initial_peak=self.p.initial_peak,
            scale_to_price=self.p.scale_to_price
        )
        self.preds, _ = gp.predict_price()

    def once(self, start, end):
        for i in range(self.data.buflen()):
            self.lines.gp_signal[i] = self.preds[i]

# -------------------------
# Strategy with Trend Gate
# -------------------------
class ReversedGriffithsCrossStrategy(bt.Strategy):
    params = dict_to_params(GRIFFITHS_DEFAULTS) + (
        ('stop_loss_pct', 0.05),
        ('max_risk_pct', 0.02),
        ('trail_stop_pct', 0.10),
    )

    def __init__(self):
        self.gp_ind = GriffithsPredictorBT(
            self.data,
            length=self.p.length,
            lower_bound=self.p.lower_bound,
            upper_bound=self.p.upper_bound,
            bars_fwd=self.p.bars_fwd,
            peak_decay=self.p.peak_decay,
            initial_peak=self.p.initial_peak,
            scale_to_price=self.p.scale_to_price
        )
        self.gp_signal = self.gp_ind.gp_signal
        self.order = None
        self.stop_order = None
        self.entry_price = None
        self.entry_date = None

    def bearish_cross(self, prev_bar, curr_bar):
        return prev_bar >= 0 and curr_bar < 0

    def bullish_cross(self, prev_bar, curr_bar):
        return prev_bar <= 0 and curr_bar > 0

    def notify_order(self, order):
        date = self.data.datetime.date(0)
        if order.status in [order.Completed]:
            if order.isbuy():
                logger.info(f"{date}: BUY EXECUTED at {order.executed.price:.2f} size {order.executed.size}")
                self.entry_date = date
            elif order.issell():
                logger.info(f"{date}: SELL EXECUTED at {order.executed.price:.2f} size {order.executed.size}")
                self.entry_date = date
            self.order = None

    def next(self):
        date = self.data.datetime.date(0)
        if self.order:
            return

        gp_val, gp_prev = self.gp_signal[0], self.gp_signal[-1] if len(self.gp_signal)>1 else 0
        if abs(gp_val) < 0.005:
            return

        port_val, price = self.broker.getvalue(), self.data.close[0]
        risk_amt = port_val * self.p.max_risk_pct
        share_risk = price * self.p.stop_loss_pct
        size = max(1, int(risk_amt / share_risk))
        size = min(size, int((port_val*0.15)/price))

        trend_flag = self.data.is_trending[0]

        if trend_flag == 1:
            if self.bearish_cross(gp_prev, gp_val) and self.data.volume[0] > self.data.volume[-5]:
                if self.position.size < 0:
                    self.cancel(self.stop_order)
                    self.order = self.close()
                else:
                    stop_pr = price*(1-self.p.stop_loss_pct)
                    self.order = self.buy(size=size)
                    self.stop_order = self.sell(size=size, exectype=bt.Order.Stop, price=stop_pr)
                    logger.info(f"{date}: LONG {size} @ {price:.2f} stop {stop_pr:.2f}")
                    self.entry_price = price
            elif self.bullish_cross(gp_prev, gp_val) and self.data.volume[0] > self.data.volume[-5]:
                if self.position.size > 0:
                    self.cancel(self.stop_order)
                    self.order = self.close()
                else:
                    stop_pr = price*(1+self.p.stop_loss_pct)
                    self.order = self.sell(size=size)
                    self.stop_order = self.buy(size=size, exectype=bt.Order.Stop, price=stop_pr)
                    logger.info(f"{date}: SHORT {size} @ {price:.2f} stop {stop_pr:.2f}")
                    self.entry_price = price
        else:
            if self.position.size >= 0:
                if self.position.size>0:
                    self.order = self.close()
                stop_pr = price*(1+self.p.stop_loss_pct)
                self.order = self.sell(size=size)
                self.stop_order = self.buy(size=size, exectype=bt.Order.Stop, price=stop_pr)

        if self.position and self.entry_price and self.stop_order:
            curr, trail = price, self.p.trail_stop_pct
            cur_stop = self.stop_order.created.price
            if self.position.size>0:
                new = curr*(1-trail)
                if new>cur_stop:
                    self.cancel(self.stop_order)
                    self.stop_order = self.sell(size=self.position.size, exectype=bt.Order.Stop, price=new)
            else:
                new = curr*(1+trail)
                if new<cur_stop:
                    self.cancel(self.stop_order)
                    self.stop_order = self.buy(size=abs(self.position.size), exectype=bt.Order.Stop, price=new)

        if self.position and self.entry_date:
            days = (date-self.entry_date).days
            if days>10:
                self.cancel(self.stop_order)
                self.order = self.close()
                logger.info(f"{date}: Time exit after {days} days")

# -------------------------
# Backtest Runner with Metrics
# -------------------------
def run_backtest(strategy_class, data_feed, symbol_name, cash=10000, commission=0.001):
    cerebro = bt.Cerebro(runonce=True, preload=True)
    cerebro.addstrategy(strategy_class)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=95)
    cerebro.adddata(data_feed)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.01)
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    logger.info(f"Running {strategy_class.__name__} Strategy...")
    result = cerebro.run()
    strat = result[0]

    sharpe    = strat.analyzers.sharpe.get_analysis()
    returns   = strat.analyzers.returns.get_analysis()
    drawdown  = strat.analyzers.drawdown.get_analysis()
    trades    = strat.analyzers.trades.get_analysis()
    max_dd_dur= drawdown.get('maxdrawdownperiod', 'N/A')

    # Logging trade stats
    logger.info(f"Returns Analysis {strategy_class.__name__}:")
    logger.info("%s", returns)
    logger.info(f"Trade Analysis {strategy_class.__name__}:")
    total_trades = trades.get('total', {}).get('total', 0)
    won_trades   = trades.get('won', {}).get('total', 0)
    lost_trades  = trades.get('lost', {}).get('total', 0)
    win_rate     = won_trades / total_trades * 100 if total_trades>0 else 0
    logger.info(f"Total Trades: {total_trades}, Won: {won_trades}, Lost: {lost_trades}, Win Rate: {win_rate:.1f}%")

    # Console output
    print(f"\nResults for {symbol_name} using {strategy_class.__name__}:")
    print(f"  Sharpe Ratio: {sharpe.get('sharperatio','N/A')}")
    print(f"  Total Return: {returns.get('rtot',0)*100:.2f}%")
    print(f"  Avg Daily Return: {returns.get('ravg',0)*100:.4f}%")
    print(f"  Avg Annual Return: {((1+returns.get('ravg',0))**252-1)*100:.2f}%")
    print(f"  Max Drawdown: {drawdown.get('drawdown',0)*100:.2f}%")
    print(f"  Max Drawdown Duration: {max_dd_dur}")
    print(f"  Total Trades: {total_trades}")
    print(f"  Win Rate: {win_rate:.1f}%")

    logger.info("Generating plot...")
    figs = cerebro.plot(style='candle')
    return figs

# -------------------------
# Main
# -------------------------
if __name__ == '__main__':
    cash, commission = 10000, 0.001
    symbol = 'SPY'
    start, end = datetime(2020,1,1), datetime.today()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file   = os.path.join(script_dir, f"{symbol}_data.xlsx")

    print(f"Loading data from: {data_file}")
    data = pd.read_excel(data_file, index_col='Date', parse_dates=True)

    detector = CycleDetector(symbol=symbol, file_path=data_file,
                             lower_bound=GRIFFITHS_DEFAULTS['lower_bound'],
                             upper_bound=GRIFFITHS_DEFAULTS['upper_bound'],
                             length=GRIFFITHS_DEFAULTS['upper_bound'],
                             window=10, stability_threshold=5)
    df_for = data.copy()
    df_for['Date'] = df_for.index
    detector.df = df_for

    close_list      = data['Close'].tolist()
    dominant_cycles = detector.detect_cycles(close_list)
    trend_labels    = detector.classify_trend(dominant_cycles)
    trend_periods   = detector.analyze_trend_periods(trend_labels)

    is_trending = pd.Series(0, index=data.index)
    for p in trend_periods:
        mask = (data.index >= p['start_date']) & (data.index <= p['end_date'])
        is_trending.loc[mask] = 1 if p['trend']=='Trending' else 0
    data['is_trending'] = is_trending

    data_feed = MyPandasData(dataname=data, fromdate=start, todate=end, timeframe=bt.TimeFrame.Minutes)

    print("\n*********************************************")
    print("***** REVERSED GRIFFITHS + CYCLE TREND *****")
    print("*********************************************")
    run_backtest(ReversedGriffithsCrossStrategy, data_feed, symbol)
