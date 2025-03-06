import numpy as np
import pandas as pd
import backtrader as bt
import logging
from datetime import datetime, timedelta
import yfinance as yf
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
# Import USI calculation functions from usi_calculation.py
from .usi_calculation import calculate_su_sd, ultimate_smoother, calculate_usi

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#################################
# USI DEFAULTS (global)
#################################
USI_DEFAULTS = {
    'period': 28,
    'smoothing_period': 4,
    'allocation': 1.0
}

def dict_to_params(d: dict) -> tuple:
    return tuple((k, v) for k, v in d.items())

#####################################
# Fetch Data Directly with yfinance
#####################################
def fetch_yfinance_data(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    try:
        df = yf.download(symbol, start=start.strftime('%Y-%m-%d'), 
                        end=end.strftime('%Y-%m-%d'), auto_adjust=True)
        df.index = pd.to_datetime(df.index)
        if df.empty:
            raise ValueError(f"No data available for {symbol}")
        logging.info(f"Fetched data for {symbol}, shape: {df.shape}")
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        logging.info(f"Adjusted columns: {list(df.columns)}")
        return df
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

#####################################
# USI Indicator wrapped for BT using imported functions
#####################################
class USIIndicatorBT(bt.Indicator):
    lines = ('usi_signal',)
    params = dict_to_params(USI_DEFAULTS)

    def __init__(self):
        self.addminperiod(self.p.period + self.p.smoothing_period)

    def next(self):
        # Fetch prices for the required period
        prices = self.data.close.get(size=self.p.period + self.p.smoothing_period)
        if len(prices) >= self.p.period + self.p.smoothing_period:
            # Use imported functions for USI calculation
            su, sd = calculate_su_sd(prices)
            usi = calculate_usi(su, sd, period=self.p.period, smoothing_period=self.p.smoothing_period)
            self.lines.usi_signal[0] = usi[-1]  # Set the latest USI value
        else:
            self.lines.usi_signal[0] = 0  # Default to 0 if insufficient data

#######################################
# Strategy with One Buy Signal at a Time
#######################################
class USICrossStrategy(bt.Strategy):
    params = dict_to_params(USI_DEFAULTS)

    def __init__(self):
        self.usi_ind = USIIndicatorBT(self.data, 
                                    period=self.p.period,
                                    smoothing_period=self.p.smoothing_period)
        self.usi_signal = self.usi_ind.usi_signal
        self.has_bought = False  # Track if we've bought without selling

    def next(self):
        current_date = self.datas[0].datetime.date(0)
        cash = self.broker.getcash()
        allocation_used = cash * self.p.allocation

        usi_val = self.usi_signal[0]
        usi_prev = self.usi_signal[-1] if len(self.usi_signal) > 1 else 0

        if not self.position and not self.has_bought:  # No position and haven't bought yet
            if usi_val > 0 and usi_prev <= 0:  # Bullish crossover
                price = self.data.close[0]
                size = int(allocation_used // price)
                self.buy(size=size)
                self.has_bought = True
                logging.info(f"{current_date}: BUY {size} shares at {price:.2f}")
        elif self.position:  # In a position
            if usi_val < 0 and usi_prev >= 0:  # Bearish crossover
                size = self.position.size
                price = self.data.close[0]
                self.sell(size=size)
                self.has_bought = False
                logging.info(f"{current_date}: SELL {size} shares at {price:.2f}")

class BuyAndHold(bt.Strategy):
    params = (('allocation', 1.0),)

    def __init__(self):
        pass

    def next(self):
        current_date = self.datas[0].datetime.date(0)
        if not self.position:
            cash = self.broker.getcash()
            price = self.data.close[0]
            size = int((cash * self.params.allocation) // price)
            self.buy(size=size)
            logging.info(f"{current_date}: BUY {size} shares at {price:.2f}")

#######################################
# Backtest Runner
#######################################
def run_backtest(strategy_class, data_feed, cash=10000, commission=0.001):
    cerebro = bt.Cerebro(runonce=True, preload=True)
    cerebro.addstrategy(strategy_class)
    cerebro.adddata(data_feed)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.01)
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

    logging.info(f"Running {strategy_class.__name__} Strategy...")
    result = cerebro.run()
    
    strat = result[0]
    sharpe = strat.analyzers.sharpe.get_analysis()
    returns = strat.analyzers.returns.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    max_drawdown_duration = drawdown.get('maxdrawdownperiod', 'N/A')

    logging.info(f"Returns Analysis {strategy_class.__name__}:")
    logging.info("\n%s", returns)

    print(f"  Sharpe Ratio: {sharpe['sharperatio']:.2f}")
    print(f"  Total Return: {returns['rtot']*100:.2f}%")
    print(f"  Avg Daily Return: {returns['ravg']*100:.2f}%")
    print(f"  Avg Annual Return: {((1+returns['ravg'])**252 - 1)*100:.2f}%")
    print(f"  Max Drawdown: {drawdown['drawdown']*100:.2f}%")
    print(f"  Max Drawdown Duration: {max_drawdown_duration}")

    logging.info("Generating plot...")
    cerebro.plot()
    plt.savefig(f"{strategy_class.__name__}_plot.png")
    logging.info(f"Plot saved as {strategy_class.__name__}_plot.png")

if __name__ == '__main__':
    cash = 10000
    commission = 0.001

    symbol = 'NVDA'
    start = datetime.now() - timedelta(days=365)
    end = datetime.now()

    data = fetch_yfinance_data(symbol=symbol, start=start, end=end)

    if data.empty:
        logging.error("No data fetched for NVDA")
        exit()

    data_feed = bt.feeds.PandasData(dataname=data, fromdate=start, todate=end)

    print("*********************************************")
    print("*************** USI CROSS *******************")
    print("*********************************************")
    run_backtest(strategy_class=USICrossStrategy, data_feed=data_feed, cash=cash, commission=commission)

    print("\n*********************************************")
    print("************* BUY AND HOLD ******************")
    print("*********************************************")
    run_backtest(strategy_class=BuyAndHold, data_feed=data_feed, cash=cash, commission=commission)