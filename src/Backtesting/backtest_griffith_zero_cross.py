from datetime import datetime
import logging
import backtrader as bt
import numpy as np
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

     in-sample predictions are stored in `gp_signal`.
    """

    lines = ('gp_signal',) 

    # After instaniaton, params are accessed as self.p.length, etc.
    params = dict_to_params(GRIFFITHS_DEFAULTS)


    def __init__(self):
        self.addminperiod(self.p.upper_bound)   # Need minimum bars before computing once()

        size = self.data.buflen() 
        predictions = np.zeros(size)

        # Instantiate predictor
        gp = GriffithsPredictor(
            close_prices=data['Close'].values,
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


    # ----------------------------------------------------------------------
    # Assign the precomputed Griffith Predictions to gp_signal
    # ----------------------------------------------------------------------
    def once(self, start, end):
        """
        'once' is called once when loading the full dataset in backtesting mode,
        so we can do a batch calculation.
        """
        # Store 'predictions' in self.lines
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

    def next(self):
        # Log the current date
        current_date = self.datas[0].datetime.date(0)

        # allocation
        cash = self.broker.getcash()
        allocation_used = cash * self.p.allocation

        # Current and previous bar's Griffiths Predictor
        gp_val = self.gp_signal[0]
        gp_prev = self.gp_signal[-1]

        # Check if we already have a position
        if not self.position:  # If not in a position
            # Buy if crossing above 0
            if gp_val > 0 and gp_prev <= 0:
                cash = self.broker.getcash()  # available cash
                price = self.data.close[0]    # current asset price
                # allocate X% of cash (self.p.allocation)
                size = int((cash * self.p.allocation) // price)

                self.buy(size=size)
                logging.info(f"{current_date}: BUY {size} shares at {price:.2f}")

        else:
            # Sell if crossing below 0
            if gp_val < 0 and gp_prev >= 0:
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

    symbol = 'SPY'
    start = datetime(2020,1,1)
    end = datetime.today()

    data = DataFetcher().get_stock_data(symbol=symbol, start_date=start, end_date=end)

    # Convert pandas DataFrame into Backtrader data feed
    data_feed = bt.feeds.PandasData(dataname=data, fromdate=start, todate=end) 


    print("*********************************************")
    print("************* Griffiths CROSS ***************")
    print("*********************************************")
    run_backtest(strategy_class=GriffithsCrossStrategy, data_feed=data_feed, cash=cash, commission=commission )



