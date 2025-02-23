from datetime import datetime
import logging
import backtrader as bt
import numpy as np
from src.Data_Retrieval.data_fetcher import DataFetcher


logging.basicConfig(level=logging.INFO, 
                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class SmaCross(bt.Strategy):
    params = (
        ('short_period', 50),
        ('long_period', 200),
        ('allocation', 1.0),  # Allocate available cash 
    )

    def __init__(self):
        self.sma_short = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.short_period)
        self.sma_long = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.long_period)
        self.crossover = bt.indicators.CrossOver(self.sma_short, self.sma_long)

    def next(self):
        # Log the current date
        current_date = self.datas[0].datetime.date(0)

        # Check if we already have a position
        if not self.position:  # If not in a position
            if self.crossover > 0:  # Buy signal
                cash = self.broker.getcash()  # Get available cash
                price = self.data.close[0]  # Current price of the asset
                size = (cash * self.params.allocation) // price  # Buy with allocated cash
                
                self.buy(size=size)  # Execute the buy order
                logging.info(f"{current_date}: BUY {size} shares at {price:.2f}")
        
        elif self.crossover < 0:  # Sell signal
            size = self.position.size  # Number of shares held
            self.sell(size=size)  # Sell the entire position
            price = self.data.close[0]  # Current price of the asset
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
    cerebro = bt.Cerebro()
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
    start = datetime(2010,1,1)
    end = datetime.today()

    data = DataFetcher().get_stock_data(symbol=symbol, start_date=start, end_date=end)

    logging.info(f"data after Yahoo Finance Call \n {data}")

    # Convert pandas DataFrame into Backtrader data feed
    data_feed = bt.feeds.PandasData(dataname=data,  # Your custom pandas DataFrame
                                    fromdate=start,
                                    todate=end) 
    print("Backtrader sees len(data_feed):", len(data_feed))

    print("*********************************************")
    print("************* SMA CROSS *********************")
    print("*********************************************")
    run_backtest(strategy_class=SmaCross, data_feed=data_feed, cash=cash, commission=commission )


    print("*********************************************")
    print("************* BUY AND HOLD ******************")
    print("*********************************************")  
    run_backtest(strategy_class=BuyAndHold, data_feed=data_feed, cash=cash, commission=commission )

