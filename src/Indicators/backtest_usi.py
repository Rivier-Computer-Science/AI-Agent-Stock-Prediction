import numpy as np
import pandas as pd
import backtrader as bt
from datetime import datetime, timedelta
import yfinance as yf

# USI Calculation Functions
def calculate_su_sd(prices):
    prices = np.asarray(prices).flatten()
    su = np.maximum(np.diff(prices, prepend=prices[0]), 0)
    sd = np.maximum(-np.diff(prices, prepend=prices[0]), 0)
    return su, sd

def ultimate_smoother(data, period):
    a1 = np.exp(-1.414 * np.pi / period)
    b1 = 2 * a1 * np.cos(1.414 * 180 / period)
    c2, c3 = b1, -a1 * a1
    c1 = (1 + c2 - c3) / 4
    smoothed = np.zeros_like(data)
    for i in range(len(data)):
        smoothed[i] = data[i] if i < 3 else (
            (1 - c1) * data[i] + (2 * c1 - c2) * data[i - 1] - (c1 + c3) * data[i - 2] + 
            c2 * smoothed[i - 1] + c3 * smoothed[i - 2]
        )
    return smoothed

def calculate_usi(prices, period=28):
    su, sd = calculate_su_sd(prices)
    usu, usd = ultimate_smoother(su, period), ultimate_smoother(sd, period)
    usi = np.zeros(len(prices))
    valid_idx = (usu + usd > 0) & (usu > 0.01) & (usd > 0.01)
    usi[valid_idx] = (usu[valid_idx] - usd[valid_idx]) / (usu[valid_idx] + usd[valid_idx])
    return usi

# Data Fetcher
class DataFetcher:
    def __init__(self, start_date: datetime = None, end_date: datetime = None):
        self.start_date = start_date or datetime.now() - timedelta(days=365)
        self.end_date = end_date or datetime.now()

    def get_stock_data(self, symbol: str) -> pd.DataFrame:
        try:
            df = yf.download(symbol, start=self.start_date.strftime('%Y-%m-%d'), 
                           end=self.end_date.strftime('%Y-%m-%d'), auto_adjust=True)
            df.index = pd.to_datetime(df.index)
            if df.empty:
                raise ValueError(f"No data available for {symbol}. Check the stock symbol.")
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()

# USI Backtrader Indicator
class USIIndicator(bt.Indicator):
    lines = ('usi_signal',)
    params = (('period', 28), ('smoothing_period', 4))

    def __init__(self):
        self.addminperiod(self.p.period + self.p.smoothing_period)

    def next(self):
        prices = self.data.close.get(size=self.p.period + self.p.smoothing_period)
        if len(prices) >= self.p.period + self.p.smoothing_period:
            usi = calculate_usi(prices, self.p.period)
            self.lines.usi_signal[0] = usi[-1]

# USI Crossing Strategy
class USICrossStrategy(bt.Strategy):
    params = (('period', 28), ('printlog', True))

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.usi_ind = USIIndicator(self.data, period=self.p.period)
        self.usi_signal = self.usi_ind.usi_signal

    def log(self, txt, dt=None):
        if self.p.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()} {txt}')

    def next(self):
        close_price = self.dataclose[0]
        usi_val = self.usi_signal[0]
        usi_prev = self.usi_signal[-1] if len(self.usi_signal) > 1 else 0

        self.log(f'Close: {close_price:.2f}, USI: {usi_val:.4f}')

        if not self.position:
            if usi_val > 0 and usi_prev <= 0:
                self.order = self.buy()
                self.log(f'BUY CREATE, {close_price:.2f}')
        elif usi_val < 0 and usi_prev >= 0:
            self.order = self.sell()
            self.log(f'SELL CREATE, {close_price:.2f}')

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}')
        self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(f'OPERATION PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}')

# Backtest Runner
def run_strategy(strategy_class, strategy_name, data_df):
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)

    # Debug: Inspect columns before processing
    print(f"Raw DataFrame columns: {list(data_df.columns)}")
    
    # Ensure columns are a flat list of strings
    data_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    print(f"Adjusted DataFrame columns: {list(data_df.columns)}")

    data = bt.feeds.PandasData(
        dataname=data_df,
        open='Open',
        high='High',
        low='Low',
        close='Close',
        volume='Volume',
        openinterest=-1
    )
    cerebro.adddata(data)
    cerebro.addstrategy(strategy_class, data_df=data_df, printlog=True)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, timeframe=bt.TimeFrame.NoTimeFrame, _name='timereturn')

    print(f'\nRunning {strategy_name}...')
    print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
    results = cerebro.run()
    strat = results[0]
    print(f'Final Portfolio Value: {cerebro.broker.getvalue():.2f}')

    sharpe = strat.analyzers.sharpe.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    timereturn = strat.analyzers.timereturn.get_analysis()

    strategy_returns = pd.Series(timereturn)
    cumulative_return = (strategy_returns + 1.0).prod() - 1.0
    start_date = data_df.index[0]
    end_date = data_df.index[-1]
    num_years = (end_date - start_date).days / 365.25
    annual_return = (1 + cumulative_return) ** (1 / num_years) - 1 if num_years != 0 else 0.0

    print(f'\n{strategy_name} Performance Metrics:')
    print('----------------------------------------')
    print(f"Sharpe Ratio: {sharpe.get('sharperatio', 'N/A')}")
    print(f"Total Return: {cumulative_return * 100:.2f}%")
    print(f"Annual Return: {annual_return * 100:.2f}%")
    print(f"Max Drawdown: {drawdown.max.drawdown:.2f}%")

    return {
        'strategy_name': strategy_name,
        'sharpe_ratio': sharpe.get('sharperatio', 'N/A'),
        'total_return': cumulative_return * 100,
        'annual_return': annual_return * 100,
        'max_drawdown': drawdown.max.drawdown,
    }

if __name__ == '__main__':
    company = 'NVDA'
    data_fetcher = DataFetcher()
    data_df = data_fetcher.get_stock_data(company)

    if data_df.empty:
        print(f"No price data found for {company}")
        exit()

    # Prepare data for Backtrader
    data_df.rename(columns={
        'Adj Close': 'Close',
        'Close': 'Close',
        'Open': 'Open',
        'High': 'High',
        'Low': 'Low',
        'Volume': 'Volume'
    }, inplace=True)
    data_df = data_df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

    # Run USI Cross Strategy
    usi_metrics = run_strategy(USICrossStrategy, 'USI Cross Strategy', data_df)