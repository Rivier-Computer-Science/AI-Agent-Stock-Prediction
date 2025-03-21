import numpy as np
import os
import pandas as pd
import backtrader as bt
import logging
from datetime import datetime
from src.Data_Retrieval.data_fetcher import DataFetcher
from src.Indicators.griffiths_predictor import GriffithsPredictor


logging.basicConfig(level=logging.INFO,
                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


#################################
# GRIFFITHS DEFAULTS (global)
#################################
GRIFFITHS_DEFAULTS = {
    'make_stationary' : False,
    'use_log_diff' : False,
    'length': 18,
    'lower_bound': 18,
    'upper_bound': 40,
    'bars_fwd': 2,
    'peak_decay': 0.991,
    'initial_peak': 0.0001,
    'scale_to_price': False,
    'allocation' : 1.0
}

def dict_to_params(d: dict) -> tuple:
    """
    Convert a dict into a Backtrader 'params' tuple,
    i.e. { 'length': 18 } -> (('length', 18), ...)
    """
    return tuple((k, v) for k, v in d.items())

#####################################
# Indicator wrapped for BT
#####################################
class GriffithsPredictorBT(bt.Indicator):
    """
    Wraps the existing indicator into a Backtrader Indicator.
    """
    lines = ('gp_signal',)
    params = dict_to_params(GRIFFITHS_DEFAULTS)

    def __init__(self):
        self.addminperiod(self.p.upper_bound)  # Ensure enough data is available

        size = len(self.data)  # Get data size
        predictions = np.zeros(size)

        close_prices = np.array(self.data.close)  # Convert to NumPy array

        # Instantiate predictor
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

        # Get the predictions
        self.preds, _ = gp.predict_price()

    def once(self, start, end):
        """
        'once' is called when loading the full dataset in backtesting mode.
        """
        for i in range(self.data.buflen()):
            self.lines.gp_signal[i] = self.preds[i]

#######################################
# Strategy
#######################################
class GriffithsCrossStrategy(bt.Strategy):
    params = dict_to_params(GRIFFITHS_DEFAULTS)

    def __init__(self):
        # Add our indicator to the data
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

        gp_val = self.gp_signal[0]
        gp_prev = self.gp_signal[-1] if len(self.gp_signal) > 1 else 0

        if self.bullish_cross(gp_prev, gp_val):
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

        elif self.bearish_cross(gp_prev, gp_val):
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


def run_backtest(strategy_class, data_feed, cash=10000, commission=0.001):
    #cerebro = bt.Cerebro()
    cerebro = bt.Cerebro(runonce=True, preload=True)
    cerebro.addstrategy(strategy_class)
    cerebro.adddata(data_feed)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission)

    # Add analyzers to the backtest
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.01)
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

    # Run the backtest
    logging.info(f"Running {strategy_class.__name__} Strategy...")
    result = cerebro.run()

    # Extract the strategy and analyzer data
    strat = result[0]
    sharpe = strat.analyzers.sharpe.get_analysis()
    returns = strat.analyzers.returns.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    max_drawdown_duration = drawdown.get('maxdrawdownperiod', 'N/A')  # Use 'N/A' if missing

    print("Sharpe Analysis:", sharpe)


    # Log the detailed analysis
    logging.info(f"Returns Analysis {strategy_class.__name__}:")
    logging.info("\n%s", returns)  # Log the whole analysis dictionary

    
    sharpe_ratio = sharpe.get('sharperatio', None)
    if sharpe_ratio is not None:
        print(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
    else:
        print("  Sharpe Ratio: N/A (insufficient data or all negative returns)")
    print(f"  Total Return: {returns['rtot']*100:.2f}%")
    print(f"  Avg Daily Return: {returns['ravg']*100:.2f}%")
    print(f"  Avg Annual Return: {((1+returns['ravg'])**252 - 1)*100:.2f}%")
    print(f"  Max Drawdown: {drawdown.drawdown*100:.2f}%")
    print(f"  Max Drawdown Duration: {max_drawdown_duration}")

    cerebro.plot()


if __name__ == '__main__':
    cash = 10000
    commission=0.001

    symbol = 'SPY'
    start = datetime(2020,1,1)
    end = datetime.today()

    # Load the data from the Excel file
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get script's directory
    data_file = os.path.join(script_dir, f"{symbol}_data.xlsx")

    print(f"Loading data from: {data_file}")

    data = pd.read_excel(data_file, index_col='Date', parse_dates=True)

    # Convert pandas DataFrame into Backtrader data feed
    data_feed = bt.feeds.PandasData(
        dataname=data,
        fromdate=start,
        todate=end,
        timeframe=bt.TimeFrame.Minutes  # Set to minute data
    )

    print("Data columns:", data.columns)

    print("*********************************************")
    print("************* Griffiths CROSS ***************")
    print("*********************************************")
    run_backtest(strategy_class=GriffithsCrossStrategy, data_feed=data_feed, cash=cash, commission=commission )

