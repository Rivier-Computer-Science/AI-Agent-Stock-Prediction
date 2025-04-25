import os
import sys
import backtrader as bt
import pandas as pd
import logging
from datetime import datetime
from dotenv import load_dotenv
from src.Data_Retrieval.data_fetcher import DataFetcher
from src.Indicators.bollinger import BollingerBands
from src.Indicators.cycle_trend_detection import CycleDetector

# Load environment variables
load_dotenv()
# Ensure USER_AGENT is set
os.environ.setdefault('USER_AGENT', 'BacktestScript/1.0')

# Logger setup
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# -------------------------
# Custom Data Feed with Trend Flag
# -------------------------
class MyPandasData(bt.feeds.PandasData):
    lines = ('is_trending',)
    params = (
        ('datetime', None),
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
        ('openinterest', None),
        ('is_trending', 'is_trending'),
    )

# -------------------------
# Strategies
# -------------------------
class BollingerCrewAIStrategy(bt.Strategy):
    params = dict(company='SPY', printlog=True)
    def __init__(self):
        self.dataclose = self.datas[0].close
        self.trend     = self.datas[0].is_trending
        # Precompute bands from full Close series
        df = pd.DataFrame({'Close': list(self.data.close.get(size=len(self.data)))},
                          index=[self.data.datetime.date(i) for i in range(len(self.data))])
        self.bands = BollingerBands(df).calculate_bands()
        # Attempt CrewAI agent (requires API key)
        try:
            from src.Agents.Bollinger_agent.bollinger_agent import BollingerAnalysisAgents
            from crewai import Crew
            agent = BollingerAnalysisAgents().bollinger_bands_investment_advisor()
            task  = BollingerAnalysisAgents().bollinger_analysis(agent, self.bands)
            self.ai_output = Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
            logger.info('CrewAI initialized')
        except Exception as e:
            logger.warning(f'CrewAI skip: {e}')
    def log(self, txt, dt=None):
        if not self.params.printlog: return
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')
    def next(self):
        price    = self.dataclose[0]
        upper    = self.bands['Upper Band'].iloc[-1]
        lower    = self.bands['Lower Band'].iloc[-1]
        trending = self.trend[0]
        self.log(f'Trend: {"T" if trending else "NT"} | U:{upper:.2f}, L:{lower:.2f}, C:{price:.2f}')
        if trending:
            # Standard Bollinger logic
            if price < lower and not self.position:
                self.order = self.buy(); self.log(f'BUY @ {price:.2f}')
            elif price > upper and self.position:
                self.order = self.sell(); self.log(f'SELL @ {price:.2f}')
        else:
            # Non-trending: force short
            if not self.position or self.position.size > 0:
                if self.position.size > 0:
                    self.order = self.close()
                self.order = self.sell(); self.log(f'SHORT @ {price:.2f}')
    def notify_order(self, order):
        if order.status == order.Completed:
            act = 'BUY' if order.isbuy() else 'SELL'
            self.log(f'{act} executed @ {order.executed.price:.2f}')
        self.order = None
    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(f'P/L gross {trade.pnl:.2f}, net {trade.pnlcomm:.2f}')

class BollingerStrategy(bt.Strategy):
    params = dict(printlog=True)
    def __init__(self):
        self.dataclose = self.datas[0].close
        self.trend     = self.datas[0].is_trending
        df = pd.DataFrame({'Close': list(self.data.close.get(size=len(self.data)))},
                          index=[self.data.datetime.date(i) for i in range(len(self.data))])
        self.bands = BollingerBands(df).calculate_bands()
    def log(self, txt, dt=None):
        if not self.params.printlog: return
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')
    def next(self):
        price    = self.dataclose[0]
        upper    = self.bands['Upper Band'].iloc[-1]
        lower    = self.bands['Lower Band'].iloc[-1]
        trending = self.trend[0]
        self.log(f'Trend: {"T" if trending else "NT"} | U:{upper:.2f}, L:{lower:.2f}, C:{price:.2f}')
        if trending:
            if price < lower and not self.position:
                self.order = self.buy(); self.log(f'BUY @ {price:.2f}')
            elif price > upper and self.position:
                self.order = self.sell(); self.log(f'SELL @ {price:.2f}')
        else:
            if not self.position or self.position.size > 0:
                if self.position.size > 0:
                    self.order = self.close()
                self.order = self.sell(); self.log(f'SHORT @ {price:.2f}')
    def notify_order(self, order):
        if order.status == order.Completed:
            act = 'BUY' if order.isbuy() else 'SELL'
            self.log(f'{act} executed @ {order.executed.price:.2f}')
        self.order = None
    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(f'P/L gross {trade.pnl:.2f}, net {trade.pnlcomm:.2f}')

# -------------------------
# Runner
# -------------------------
def run_strategy(cls, name, data_df=None, file_path=None, company=None):
    # Load from xlsx or fetch
    if data_df is None or getattr(data_df, 'empty', False):
        if file_path and os.path.exists(file_path):
            data_df = pd.read_excel(file_path, index_col='Date', parse_dates=True)
            logger.info(f'Loaded from {file_path}')
        else:
            logger.info(f'Fetching data for {company}')
            data_df = DataFetcher(start_date=datetime(2015,1,1), end_date=datetime(2024,10,30)).get_stock_data(company)
    if data_df is None or data_df.empty:
        logger.error(f'No data for {name}')
        return None
    # Normalize
    if isinstance(data_df.columns, pd.MultiIndex):
        data_df.columns = [c[0] for c in data_df.columns]
    data_df.rename(columns={'Adj Close':'Close'}, inplace=True)
    data_df = data_df[['Open','High','Low','Close','Volume']].dropna()
    # Trend detection
    detector = CycleDetector(lower_bound=18, upper_bound=40, length=40, window=10, stability_threshold=5)
    df_for = data_df.copy()
    df_for['Date'] = df_for.index
    detector.df = df_for
    close_list = data_df['Close'].values.tolist()
    labels = detector.classify_trend(detector.detect_cycles(close_list))
    periods = detector.analyze_trend_periods(labels) if labels else []
    flag = pd.Series(0, index=data_df.index)
    for p in periods:
        mask = (data_df.index >= p['start_date']) & (data_df.index <= p['end_date'])
        flag.loc[mask] = 1 if p['trend'] == 'Trending' else 0
    data_df['is_trending'] = flag
    # Prepare feed
    feed = MyPandasData(dataname=data_df)
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.adddata(feed)
    cerebro.addstrategy(cls, company=company or 'SPY', printlog=True)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, timeframe=bt.TimeFrame.NoTimeFrame, _name='timereturn')
    print(f'\nRunning {name}...')
    print(f'Starting portfolio: {cerebro.broker.getvalue():.2f}')
    results = cerebro.run()
    strat = results[0]
    print(f'Final portfolio: {cerebro.broker.getvalue():.2f}')
    # Metrics
    sharpe = strat.analyzers.sharpe.get_analysis()
    dd = strat.analyzers.drawdown.get_analysis()
    tr = strat.analyzers.timereturn.get_analysis()
    ret_ser = pd.Series(tr)
    cum = (ret_ser + 1.0).prod() - 1.0
    yrs = (data_df.index[-1] - data_df.index[0]).days / 365.25
    ann = (1+cum)**(1/yrs)-1 if yrs else 0
    print(f'\n{name} metrics: Sharpe={sharpe.get("sharperatio","N/A")}, Total={cum*100:.2f}%, Annual={ann*100:.2f}%, MaxDD={dd.max.drawdown:.2f}%')
    return {'name': name, 'sharpe': sharpe.get('sharperatio','N/A'), 'total_return': cum*100, 'annual_return': ann*100, 'max_drawdown': dd.max.drawdown}

# -------------------------
# Main
# -------------------------
if __name__ == '__main__':
    company = 'SPY'
    xf = f"{company}_data.xlsx"
    df = None
    fp = xf if os.path.exists(xf) else None
    if fp is None:
        df = DataFetcher(start_date=datetime(2015,1,1), end_date=datetime(2024,10,30)).get_stock_data(company)
    metrics = []
    for cls, name in [(BollingerCrewAIStrategy, 'CrewAI Bollinger'), (BollingerStrategy, 'Non-Crew Bollinger')]:
        m = run_strategy(cls, name, data_df=df, file_path=fp, company=company)
        if m:
            metrics.append(m)
    if metrics:
        comp = pd.DataFrame(metrics)
        print("\nComparison:")
        print(comp.to_string(index=False))
