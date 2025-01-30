import pandas as pd
import yfinance as yf
import math
from datetime import datetime, timedelta

class DataFetcher:
    def __init__(self, symbol, start_date=None, end_date=None):
        self.symbol = symbol
        self.start_date = start_date if start_date else (datetime.today() - timedelta(days=60)).strftime('%Y-%m-%d')
        self.end_date = end_date if end_date else datetime.today().strftime('%Y-%m-%d')

    def get_stock_data(self):
        df = yf.download(self.symbol, start=self.start_date, end=self.end_date)
        return df['Close']

def super_smoother(price_series, period):
    a1 = math.exp(-1.414 * math.pi / period)
    b1 = 2 * a1 * math.cos(1.414 * 180 / period)
    c1 = 1 - b1 + a1 * a1
    c2 = b1
    c3 = -a1 * a1

    smooth_series = [0] * len(price_series)
    smooth_series[0] = price_series[0]
    smooth_series[1] = price_series[1]

    for i in range(2, len(price_series)):
        smooth_series[i] = (
            c1 * (price_series[i] + price_series[i - 1]) / 2
            + c2 * smooth_series[i - 1]
            + c3 * smooth_series[i - 2]
        )
    return smooth_series
"""
#Usage example:
symbol = 'ES=F'
fetcher = DataFetcher(symbol=symbol)
price_data = fetcher.get_stock_data()
smoothed_prices = super_smoother(price_data.squeeze().tolist(), 14)
smoothed_df = pd.DataFrame(smoothed_prices, index=price_data.index, columns=['Smoothed'])
print(smoothed_df[:10])
"""

