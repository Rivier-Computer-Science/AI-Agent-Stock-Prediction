import backtrader as bt
from datetime import datetime
import pandas as pd
import yfinance as yf
import logging
import os
from dotenv import load_dotenv

# Load environment variables (e.g., for API keys if needed)
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class TimingTradingSystemStrategy(bt.Strategy):
    params = dict(
        stock='AAPL',
        printlog=False,
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.stock = self.params.stock
        self.decision = None

    def fetch_earnings_date(self):
        # Simulate fetching earnings dates
        # In a real implementation, integrate with an API or data source
        return datetime(2024, 10, 31)  # Example fixed date

    def analyze_sentiment(self, earnings_date):
        # Placeholder for sentiment analysis around the earnings date
        days_until_earnings = (earnings_date - datetime.now()).days
        logging.debug(f"Days until earnings: {days_until_earnings}")
        if earnings_date and days_until_earnings <= 5:
            return "buy"
        else:
            return "sell"

    def next(self):
        # Run sentiment analysis logic to get buy/sell decision
        if not self.position:
            earnings_date = self.fetch_earnings_date()
            self.decision = self.analyze_sentiment(earnings_date)

            # Execute buy or sell decision
            if self.decision == 'buy':
                cash = self.broker.getcash()
                size = int(cash // self.dataclose[0])
                if size > 0:
                    self.order = self.buy(size=size)
                    if self.params.printlog:
                        self.log(f'BUY CREATE, {self.dataclose[0]:.2f}')
            elif self.decision == 'sell' and self.position:
                self.order = self.sell(size=self.position.size)
                if self.params.printlog:
                    self.log(f'SELL CREATE, {self.dataclose[0]:.2f}')

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}')
            self.bar_executed = len(self)
        self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(f'OPERATION PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}')


class BuyAndHold(bt.Strategy):
    params = (
        ('allocation', 1.0),  # Allocate 100% of available cash to buy and hold
    )

    def __init__(self):
        pass  # No indicators needed for Buy-and-Hold

    def next(self):
        current_date = self.datas[0].datetime.date(0)
        # Buy once and hold
        if not self.position:
            cash = self.broker.getcash()  # Get available cash
            price = self.data.close[0]  # Current price
            size = int((cash * self.params.allocation) // price)  # Calculate size to buy
            if size > 0:
                self.buy(size=size)  # Execute buy order
                logging.info(f"{current_date}: BUY {size} shares at {price:.2f}")

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                logging.info(f'BUY EXECUTED, Price: {order.executed.price:.2f}')
            elif order.issell():
                logging.info(f'SELL EXECUTED, Price: {order.executed.price:.2f}')


def run_strategy(strategy_class, strategy_name, data_df, stock=None):
    """
    Runs a Backtrader strategy and returns performance metrics.
    """
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)

    # Ensure data_df has the required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_columns:
        if col not in data_df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Create the data feed with explicit column mappings
    data = bt.feeds.PandasData(
        dataname=data_df,
        datetime=None,  # Use index as datetime
        open='Open',
        high='High',
        low='Low',
        close='Close',
        volume='Volume',
        openinterest=-1
    )
    cerebro.adddata(data)

    # Add strategy with parameters
    if strategy_class == TimingTradingSystemStrategy and stock:
        cerebro.addstrategy(strategy_class, stock=stock, printlog=True)
    else:
        cerebro.addstrategy(strategy_class)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, timeframe=bt.TimeFrame.NoTimeFrame, _name='timereturn')

    print(f'\nRunning {strategy_name}...')
    print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
    results = cerebro.run()
    strat = results[0]
    print(f'Final Portfolio Value: {cerebro.broker.getvalue():.2f}')

    # Retrieve analyzers
    sharpe = strat.analyzers.sharpe.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    timereturn = strat.analyzers.timereturn.get_analysis()

    # Calculate performance metrics
    strategy_returns = pd.Series(timereturn)
    cumulative_return = (strategy_returns + 1.0).prod() - 1.0

    start_date = data_df.index[0]
    end_date = data_df.index[-1]
    num_years = (end_date - start_date).days / 365.25
    annual_return = (1 + cumulative_return) ** (1 / num_years) - 1 if num_years else 0.0

    print(f'\n{strategy_name} Performance Metrics:')
    print('----------------------------------------')
    print(f"Sharpe Ratio: {sharpe.get('sharperatio', 'N/A')}")
    print(f"Total Return: {cumulative_return * 100:.2f}%")
    print(f"Annual Return: {annual_return * 100:.2f}%")
    print(f"Max Drawdown: {drawdown.max.drawdown:.2f}%")

    # Uncomment the following line to plot the strategy results
    # cerebro.plot(style='candlestick')

    return {
        'strategy_name': strategy_name,
        'sharpe_ratio': sharpe.get('sharperatio', 'N/A'),
        'total_return': cumulative_return * 100,
        'annual_return': annual_return * 100,
        'max_drawdown': drawdown.max.drawdown,
    }


def main():
    stock = 'AAPL'
    data_df = yf.download(stock, start='2020-01-01', end='2024-10-30')

    if data_df.empty:
        print(f"No price data found for {stock}")
        sys.exit()

    # Flatten the columns if they are MultiIndex
    if isinstance(data_df.columns, pd.MultiIndex):
        data_df.columns = [' '.join(col).strip() for col in data_df.columns.values]
        # Remove the ticker symbol from the column names
        data_df.columns = [col.split(' ')[0] for col in data_df.columns]

    # If you prefer to use 'Adj Close' as 'Close', rename it
    if 'Adj Close' in data_df.columns:
        data_df['Close'] = data_df['Adj Close']

    # Remove any unnecessary columns and ensure required columns are present
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    data_df = data_df[required_columns]

    # Run TimingTradingSystemStrategy
    try:
        timing_trading_system_metrics = run_strategy(
            TimingTradingSystemStrategy,
            'Timing Trading System Strategy',
            data_df.copy(),
            stock
        )
    except Exception as e:
        print(f"Error running Timing Trading System Strategy: {e}")
        timing_trading_system_metrics = None

    # Run BuyAndHold Strategy
    try:
        buy_and_hold_metrics = run_strategy(
            BuyAndHold,
            'Buy and Hold Strategy',
            data_df.copy()
        )
    except Exception as e:
        print(f"Error running Buy and Hold Strategy: {e}")
        buy_and_hold_metrics = None

    # Print comparison of performance metrics
    print("\nPerformance Comparison:")
    print("----------------------------------------")
    if timing_trading_system_metrics and buy_and_hold_metrics:
        metrics = ['strategy_name', 'sharpe_ratio', 'total_return', 'annual_return', 'max_drawdown']
        df_metrics = pd.DataFrame([timing_trading_system_metrics, buy_and_hold_metrics], columns=metrics)
        print(df_metrics.to_string(index=False))
    else:
        print("One or both strategies failed to run. Please check the logs for more details.")


if __name__ == '__main__':
    main()
