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
from .usi_jg import calculate_usi

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
        self.addminperiod(USI_DEFAULTS['period'] + 2*USI_DEFAULTS['smoothing_period'])

        size = self.data.buflen() 
        predictions = np.zeros(size)

        self.usi_df = calculate_usi(
            df=data,
            length=self.p.period,
            window=self.p.smoothing_period
        )

        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     print("usi_values\n", self.usi_df)


    # Assign indicator values to backtrader
    def once(self, start, end):
        for i in range(self.data.buflen()):
            self.lines.usi_signal[i] = self.usi_df[i] 



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
        self.order = None
        self.pending_entry = None

    def bullish_cross(self, prev_bar, current_bar):
        return prev_bar < 0 and current_bar >= 0

    def bearish_cross(self, prev_bar, current_bar):
        return prev_bar > 0 and current_bar <= 0

    def log_position(self):
        pos_size = self.position.size if self.position else 0
        pos_type = 'NONE'
        if pos_size > 0:
            pos_type = 'LONG'
        elif pos_size < 0:
            pos_type = 'SHORT'
        logging.info(f"{self.data.datetime.date(0)}: POSITION UPDATE: {pos_type} {pos_size} shares")

    def notify_order(self, order):
        date = self.data.datetime.date(0)
        if order.status in [order.Completed]:
            if order.isbuy():
                logging.info(f"{date}: BUY EXECUTED, Price: {order.executed.price:.2f}, Size: {order.executed.size}")
            elif order.issell():
                logging.info(f"{date}: SELL EXECUTED, Price: {order.executed.price:.2f}, Size: {order.executed.size}")

            self.log_position()

            # Enter pending position after close executes
            if self.pending_entry:
                cash = self.broker.getcash()
                price = self.data.close[0]
                size = int((cash / price) * 0.95)
                if size > 0:
                    if self.pending_entry == 'LONG':
                        self.order = self.buy(size=size)
                        logging.info(f"{date}: BUY {size} shares at {price:.2f}")
                    elif self.pending_entry == 'SHORT':
                        self.order = self.sell(size=size)
                        logging.info(f"{date}: SELL {size} shares at {price:.2f}")
                self.pending_entry = None

            self.log_position()

        elif order.status in [order.Margin, order.Rejected]:
            logging.warning(f"{self.data.datetime.date(0)}: Order Failed - Margin/Rejected")
            self.order = None
            self.pending_entry = None

        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None

    def next(self):
        date = self.data.datetime.date(0)
        if self.order:
            return  # Wait for pending order to complete

        usi_val = self.usi_signal[0]
        usi_prev = self.usi_signal[-1] if len(self.usi_signal) > 1 else 0

        if self.bullish_cross(usi_prev, usi_val):
            if self.position:
                if self.position.size < 0:  # Short position active
                    logging.info(f"{date}: CLOSING SHORT POSITION BEFORE GOING LONG")
                    self.order = self.close()
                    self.pending_entry = 'LONG'
            else:
                size = int((self.broker.getcash() / self.data.close[0]) * 0.95)
                if size > 0:
                    self.order = self.buy(size=size)
                    logging.info(f"{date}: BUY {size} shares at {self.data.close[0]:.2f}")

        elif self.bearish_cross(usi_prev := self.usi_signal[-1], self.usi_signal[0]):
            if self.position:
                if self.position.size > 0:  # Long position active
                    logging.info(f"{date}: CLOSING LONG POSITION BEFORE GOING SHORT")
                    self.order = self.close()
                    self.pending_entry = 'SHORT'
            else:
                size = int((self.broker.getcash() / self.data.close[0]) * 0.95)
                if size > 0:
                    self.order = self.sell(size=size)
                    logging.info(f"{date}: SELL {size} shares at {self.data.close[0]:.2f}")


class BuyAndHold(bt.Strategy):
    params = (
        ('allocation', 1.0),  # Allocate 100% of the available cash to buy and hold (adjust as needed)
    )

    def __init__(self):
        pass  # No need for indicators in Buy-and-Hold strategy

    def next(self):
        current_date = self.datas[0].datetime.date(0)
        # Check if we already have a position (buy once and hold)
        if not self.position:  # If not in a position
            cash = self.broker.getcash()  # Get available cash
            price = self.data.close[0]  # Current price of the asset
            size = (cash * self.params.allocation) // price  # Buy with the allocated cash
            self.buy(size=size)  # Execute the buy order with calculated size
            logging.info(f"{current_date}: BUY {size} shares at {price:.2f}")

#######################################
# Backtest Runner
#######################################
def run_backtest(strategy_class, data_feed, cash=10000, commission=0.001):
    cerebro = bt.Cerebro(runonce=True, preload=True)
    cerebro.addstrategy(strategy_class)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=95) #prevent partial fills
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
    #plt.savefig(f"{strategy_class.__name__}_plot.png")
    #logging.info(f"Plot saved as {strategy_class.__name__}_plot.png")

if __name__ == '__main__':
    cash = 10000
    commission = 0.001

    symbol = 'SPY'
    start = datetime.now() - timedelta(days=365)
    end = datetime.now()

    data = DataFetcher().get_stock_data(symbol=symbol, start_date=start, end_date=end).resample('D').last().dropna()

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