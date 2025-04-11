from datetime import datetime
import logging
import backtrader as bt
import numpy as np
import datetime as dt
import ast

from src.Data_Retrieval.data_fetcher import DataFetcher  
from src.Data_Retrieval.data_fetcher import DataFetcher
from src.Agents.Research.bollinger_crew import BollingerCrew


logging.basicConfig(level=logging.INFO, 
                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


#################################
# BOLLINGER DEFAULTS (global)
#################################
symbol = 'NVDA'

BOLLINGER_DEFAULTS = {
    'ticker': symbol,
    'length': 20,
    'std': 2.0,
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
        indicator_dict = ast.literal_eval(indicator_output)
        print("*******************************************************\n\n" )
        #list(indicator_values.values())

        if isinstance(indicator_dict, dict):
            print("Object is a dictionary.")
        else:
            print(f"Object is not a dictionary. It is of type: {type(indicator_dict).__name__}")

        self.preds =  indicator_dict.values()       


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

        self.bollinger_signal = self.bollinger_ind.bollinger_signal

    def next(self):
        # Log the current date
        current_date = self.datas[0].datetime.date(0)

        # allocation
        cash = self.broker.getcash()
        allocation_used = cash * self.p.allocation

        # Current and previous bar's Griffiths Predictor
        bb_val = self.bollinger_signal[0]

        # Check if we already have a position
        if not self.position:  # If not in a position
            # Buy if crossing above 0
            if bb_val == "BUY":
                cash = self.broker.getcash()  # available cash
                price = self.data.close[0]    # current asset price
                # allocate X% of cash (self.p.allocation)
                size = int((cash * self.p.allocation) // price)

                self.buy(size=size)
                logging.info(f"{current_date}: BUY {size} shares at {price:.2f}")

        else:
            # Sell if crossing below 0
            if bb_val == "SELL":
                size = self.position.size
                price = self.data.close[0]

                self.sell(size=size)
                logging.info(f"{current_date}: SELL {size} shares at {price:.2f}")



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

    print(f"  Sharpe Ratio: {sharpe['sharperatio']:.2f}")
    print(f"  Total Return: {returns['rtot']*100:.2f}%")
    print(f"  Avg Daily Return: {returns['ravg']*100:.2f}%")
    print(f"  Avg Annual Return: {((1+returns['ravg'])**252 - 1)*100:.2f}%")
    print(f"  Max Drawdown: {drawdown.drawdown*100:.2f}%")
    print(f"  Max Drawdown Duration: {max_drawdown_duration}")

    cerebro.plot()



if __name__ == '__main__':
    cash = 10000
    commission=0.001

    start = datetime(2020,1,1)
    end = datetime.today()

    # symbol is global data because the bt.Indicator needs it
    data = DataFetcher().get_stock_data(symbol=symbol, start_date=start, end_date=end)

    # Convert pandas DataFrame into Backtrader data feed
    data_feed = bt.feeds.PandasData(dataname=data, fromdate=start, todate=end) 


    print("*********************************************")
    print("************* Bollinger CrewAI **************")
    print("*********************************************")
    run_backtest(strategy_class=BollingerCrewAIStrategy, data_feed=data_feed, cash=cash, commission=commission )