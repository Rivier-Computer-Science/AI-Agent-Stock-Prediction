from datetime import datetime
import logging
import backtrader as bt
import numpy as np
import datetime as dt
import json

from src.Data_Retrieval.data_fetcher import DataFetcher
from src.Agents.Research.bollinger_crew import BollingerCrew


# Reset logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO, 
                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


#################################
# BOLLINGER DEFAULTS (global)
#################################
symbol = 'NVDA'

BOLLINGER_DEFAULTS = {
    'ticker': symbol,
    'length': 20,
    'std': 1.0,
    'allocation' : 1.0
}


def dict_to_params(d: dict) -> tuple:
    """
    Convert a dict into a Backtrader 'params' tuple,
    i.e. { 'length': 20 } -> (('length', 20), ...)
    """
    return tuple((k, v) for k, v in d.items())

#####################################
# Indicator wrapped for BT
#####################################
class BollingerIndicatorBT(bt.Indicator):
    """
    Wraps the existing indicator into a Backtrader Indicator.

     in-sample predictions are stored in `bollinger_signal`.
    """

    lines = ('bollinger_signal',) 

    # After instaniaton, params are accessed as self.p.length, etc.
    params = dict_to_params(BOLLINGER_DEFAULTS)


    def __init__(self):
        self.addminperiod(self.p.length)   # Need minimum bars before computing once()

        size = self.data.buflen() 
        predictions = np.zeros(size)

        # Instantiate predictor
        bb = BollingerCrew(
            ticker=self.p.ticker,
            stock_data=data,
            length=self.p.length,
            std=self.p.std            
        )


        # Get the predictions
        indicator_output, _ = bb.run()
        indicator_dict = json.loads(indicator_output)
        if not isinstance(indicator_dict, dict):
            raise ValueError(f"Invalid parameter type of indicator_dict is not a dict.  It is of type: {type(indicator_dict).__name__}")

        # Dictionary will be:  {'output': {'2025-01-13': 'BUY', '2025-01-14': 'SELL', ...
        # Extract the signals (BUY, HOLD, SELL)        
        signals_dict = indicator_dict['output']
 
        # Map signal to numeric values
        signal_mapping = {"BUY": 1, "SELL": -1, "HOLD": 0}
        numeric_signals_dict = {date: signal_mapping.get(signal, None) for date, signal in signals_dict.items()}

        numeric_signals_list = list(numeric_signals_dict.values())
        print("\n\n Numeric Signals\n", numeric_signals_list)
        self.preds =  numeric_signals_list


    # ----------------------------------------------------------------------
    # Assign the precomputed Bollinger Signals to bollinger_signal
    # ----------------------------------------------------------------------
    def once(self, start, end):
        """
        'once' is called once when loading the full dataset in backtesting mode,
        so we can do a batch calculation.
        """
        # Store 'predictions' in self.lines
        for i in range(self.data.buflen()):
            self.lines.bollinger_signal[i] = self.preds[i]

 
#######################################
# Strategy
#######################################
class BollingerCrewAIStrategy(bt.Strategy):
    params = dict_to_params(BOLLINGER_DEFAULTS)

    def __init__(self):
        # Add our indicator to the data
        self.bollinger_ind = BollingerIndicatorBT(
            self.data,
            ticker=self.p.ticker,            
            length=self.p.length,
            std=self.p.std
        )
        # Print bollinger bands on chart. Not used for trading
        self.bbands = bt.indicators.BollingerBands(self.datas[0], period=self.p.length, devfactor=self.p.std)

        # Bollinger signals from CrewAI used for trading
        self.bb_signal = self.bollinger_ind.bollinger_signal
        self.order = None
        self.pending_entry = None

    def bullish_cross(self, prev_bar, current_bar):
        return prev_bar == 0 and current_bar == 1

    def bearish_cross(self, prev_bar, current_bar):
        return prev_bar == 0 and current_bar == -1

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

        bb_val = self.bb_signal[0]
        bb_prev = self.bb_signal[-1] if len(self.bb_signal) > 1 else 0

        if self.bullish_cross(bb_prev, bb_val):
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

        elif self.bearish_cross(bb_prev, bb_val):
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
 
    # Log the detailed analysis
    logging.info(f"Returns Analysis {strategy_class.__name__}:")
    logging.info("\n%s", returns)  # Log the whole analysis dictionary

    # Sharpe Ratio
    sharpe_ratio = sharpe.get('sharperatio')
    if sharpe_ratio is not None:
        print(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
    else:
        print("  Sharpe Ratio: Not available")

    # Total Return
    total_return = returns.get('rtot')
    if total_return is not None:
        print(f"  Total Return: {total_return * 100:.2f}%")
    else:
        print("  Total Return: Not available")

    # Average Daily Return
    avg_daily_return = returns.get('ravg')
    if avg_daily_return is not None:
        print(f"  Avg Daily Return: {avg_daily_return * 100:.2f}%")
    else:
        print("  Avg Daily Return: Not available")

    # Average Annual Return
    if avg_daily_return is not None:
        avg_annual_return = ((1 + avg_daily_return)**252 - 1) * 100
        print(f"  Avg Annual Return: {avg_annual_return:.2f}%")
    else:
        print("  Avg Annual Return: Not available")

    # Max Drawdown
    max_dd = getattr(drawdown, 'drawdown', None)
    if max_dd is not None:
        print(f"  Max Drawdown: {max_dd * 100:.2f}%")
    else:
        print("  Max Drawdown: Not available")

    # Max Drawdown Duration
    if max_drawdown_duration is not None:
        print(f"  Max Drawdown Duration: {max_drawdown_duration}")
    else:
        print("  Max Drawdown Duration: Not available")


    cerebro.plot()



if __name__ == '__main__':
    cash = 10000
    commission=0.001

    today = dt.datetime.today()
    #start_date = dt.datetime(2014, 1, 1)
    start = today - dt.timedelta(days=90)  # make sure inclusive
    end = today        

    # symbol is global data because the bt.Indicator needs it
    data = DataFetcher().get_stock_data(symbol=symbol, start_date=start, end_date=end)

    # Convert pandas DataFrame into Backtrader data feed
    data_feed = bt.feeds.PandasData(dataname=data, fromdate=start, todate=end) 


    print("*********************************************")
    print("************* Bollinger CrewAI **************")
    print("*********************************************")
    run_backtest(strategy_class=BollingerCrewAIStrategy, data_feed=data_feed, cash=cash, commission=commission )