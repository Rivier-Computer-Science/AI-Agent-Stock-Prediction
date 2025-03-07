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
#from .usi_calculation import calculate_su_sd, ultimate_smoother, calculate_usi
from src.Data_Retrieval.data_fetcher import DataFetcher
from .usi_trendvsswing import calculate_usi

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
# USI Indicator wrapped for BT using imported functions
#####################################
class USIIndicatorBT(bt.Indicator):
    lines = ('usi_signal',)
    params = dict_to_params(USI_DEFAULTS)

    def __init__(self):
        self.addminperiod(self.p.period + self.p.smoothing_period)

        size = self.data.buflen() 
        predictions = np.zeros(size)

        self.usi_values = calculate_usi(
            prices=data['Close'].values,
            period=self.p.period,
            smoothing_period=self.p.smoothing_period
        )


    # Assign indicator values to backtrader
    def once(self, start, end):
        for i in range(self.data.buflen()):
            self.lines.usi_signal[i] = self.usi_values[i] 

    # def next(self):
    #     # Fetch prices for the required period
    #     prices = self.data.close.get(size=self.p.period + self.p.smoothing_period)
    #     if len(prices) >= self.p.period + self.p.smoothing_period:
    #         # Use imported functions for USI calculation
    #         su, sd = calculate_su_sd(prices)
    #         usi = calculate_usi(su, sd, period=self.p.period, smoothing_period=self.p.smoothing_period)
    #         self.lines.usi_signal[0] = usi[-1]  # Set the latest USI value
    #     else:
    #         self.lines.usi_signal[0] = 0  # Default to 0 if insufficient data







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
        self.is_long = False  
        self.is_short = False

    def bullish_cross(self, prev_bar, current_bar):
        if prev_bar < 0 and current_bar > 0:
            return True
        
        return False

    def bearish_cross(self, prev_bar, current_bar):
        if prev_bar > 0 and current_bar < 0:
            return True
        
        return False

    def buy_all(self):
        if self.is_long: return

        if self.is_short:
            self.close()

        price = self.data.close[0]
        cash = self.broker.getcash()
        size = int(cash/price)

        if size > 0:
            current_date = self.datas[0].datetime.date(0)
            self.buy(size=size)
            logging.info(f"{current_date}: BUY {size} shares at {price:.2f}")
            self.is_long = True
            self.is_short = False


    def sell_all(self):
        if self.is_short: return

        if self.is_long:
            self.close()

        price = self.data.close[0]
        cash = self.broker.getcash()
        size = int(cash/price)

        if size > 0:
            current_date = self.datas[0].datetime.date(0)
            self.sell(size=size)
            logging.info(f"{current_date}: SELL {size} shares at {price:.2f}")
            self.is_short = True
            self.is_long = False


    def next(self):        
        usi_val = self.usi_signal[0]
        usi_prev = self.usi_signal[-1] if len(self.usi_signal) > 1 else 0

        if self.bullish_cross(usi_prev, usi_val):
            self.buy_all()

        if self.bearish_cross(usi_prev, usi_val):
            self.sell_all()
                

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

    symbol = 'SPY'
    start = datetime.now() - timedelta(days=365)
    end = datetime.now()

    data = DataFetcher().get_stock_data(symbol=symbol, start_date=start, end_date=end)

    if data.empty:
        logging.error(f"No data fetched for {symbol}")
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