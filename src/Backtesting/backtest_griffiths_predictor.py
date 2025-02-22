import backtrader as bt
import pandas as pd
import numpy as np
from src.Indicators.Griffiths_predictor import griffiths_predictor
from src.Indicators.High_pass_filter_function import highpass_filter
from src.Indicators.SuperSmoother_filter_function import super_smoother
from src.Data_Retrieval.data_fetcher import DataFetcher
from datetime import datetime

class GriffithsPredictorStrategy(bt.Strategy):
    params = dict(
        data_df=None,         
        printlog=True,
        length=18,        
        lower_bound=18,
        upper_bound=40,
        bars_fwd=2,       
        threshold=0.0     
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None

        data_df = self.params.data_df
        if data_df is None or data_df.empty:
            raise ValueError("No data_df provided to GriffithsPredictorStrategy.")

        self.close_prices = data_df['Close'].tolist()

        self.predictions, self.future_signals = griffiths_predictor(
            close_prices=self.close_prices,
            length=self.params.length,
            lower_bound=self.params.lower_bound,
            upper_bound=self.params.upper_bound,
            bars_fwd=self.params.bars_fwd
        )
    def next(self):
        t = len(self) - 1  
        close_price = self.dataclose[0]

        if t < len(self.predictions):
            predicted_value = self.predictions[t]
        else:
            return

        threshold = self.params.threshold
        diff = predicted_value - close_price

        if not self.position:
            if diff > threshold:
                self.order = self.buy() 
                self.log(f'BUY CREATE, close={close_price:.2f}, pred={predicted_value:.2f}')
            elif diff < -threshold:
                self.order = self.sell()
                self.log(f'SELL CREATE, close={close_price:.2f}, pred={predicted_value:.2f}')
        else:
            if self.position.size > 0 and diff < 0:
                self.close()
                self.log(f'CLOSE LONG, close={close_price:.2f}, pred={predicted_value:.2f}')
            elif self.position.size < 0 and diff > 0:
                self.close()
                self.log(f'CLOSE SHORT, close={close_price:.2f}, pred={predicted_value:.2f}')

    def log(self, txt, dt=None):
        """ Logging function """
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()} {txt}')

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

def run_griffiths_strategy(strategy_class, strategy_name, data_df, company=None):
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)

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
    cerebro.addstrategy(
        GriffithsPredictorStrategy,
        data_df=data_df,
        printlog=True,
        length=18,
        lower_bound=18,
        upper_bound=40,
        bars_fwd=2,
        threshold=0.0  
    )

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, timeframe=bt.TimeFrame.NoTimeFrame, _name='timereturn')

    print('\nRunning Griffiths Predictor Strategy...')
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

    print('\nGriffiths Predictor Performance Metrics:')
    print('----------------------------------------')
    print(f"Sharpe Ratio: {sharpe.get('sharperatio', 'N/A')}")
    print(f"Total Return: {cumulative_return * 100:.2f}%")
    print(f"Annual Return: {annual_return * 100:.2f}%")
    print(f"Max Drawdown: {drawdown.max.drawdown:.2f}%")

    metrics = {
        'strategy_name': 'Griffiths Predictor Strategy',
        'sharpe_ratio': sharpe.get('sharperatio', 'N/A'),
        'total_return': cumulative_return * 100,
        'annual_return': annual_return * 100,
        'max_drawdown': drawdown.max.drawdown,
    }
    return metrics

if __name__ == '__main__':
    company = 'NVDA'
    data_fetcher = DataFetcher(start_date=datetime(2015, 1, 1), end_date=datetime(2024, 12, 31))
    data_df = data_fetcher.get_stock_data(company)
    #print("DataFrame columns:", data_df.columns)
    #print("Head of DF:\n", data_df.head())

    if isinstance(data_df.columns, pd.MultiIndex):
        data_df.columns = [col[0] for col in data_df.columns]

    rename_map = {
        'Open': 'Open',
        'High': 'High',
        'Low': 'Low',
        'Adj Close': 'Close',
        'Close': 'Close',
        'Volume': 'Volume'
    }
    data_df.rename(columns=rename_map, inplace=True, errors='ignore')

    data_df = data_df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

    griffiths_metrics = run_griffiths_strategy(
        GriffithsPredictorStrategy,
        'Griffiths Predictor Strategy',
        data_df,
        company
    )

    print("\nGriffiths Predictor results:")
    print(griffiths_metrics)
