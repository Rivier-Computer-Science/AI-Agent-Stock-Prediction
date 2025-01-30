import math
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
#import plotly.graph_objects as go

class DataFetcher:
    def __init__(self, start_date: datetime = None, end_date: datetime = None):
        if start_date is None:
            self.start_date = datetime.today() - timedelta(days=60)
        else:
            self.start_date = start_date

        if end_date is None:
            self.end_date = datetime.today()

    def get_stock_data(self, symbol: str, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        df = yf.download(symbol, start=start_date_str, end=end_date_str)
        df.index = pd.to_datetime(df.index)
        return df


def highpass_filter(price_series, period):
    """
    Implements a highpass filter.
    Args:
        price_series (list): A time series of price data.
        period (int): Period of the filter.
    Returns:
        list: Highpass filtered series.
    """
    a1 = math.exp(-1.414 * math.pi / period)
    b1 = 2 * a1 * math.cos(1.414 * 180 / period)
    c1 = (1 + b1) / 4
    c2 = b1
    c3 = -a1 * a1

    highpass_series = [0] * len(price_series)

    for i in range(2, len(price_series)):
        highpass_series[i] = (
            c1 * (price_series[i] - 2 * price_series[i - 1] + price_series[i - 2])
            + c2 * highpass_series[i - 1]
            + c3 * highpass_series[i - 2]
        )
    return highpass_series
"""
#Usage example:
fetcher = DataFetcher()  # Initialize DataFetcher with default dates
data = fetcher.get_stock_data('ES=F')  # Fetch Apple stock data

if not data.empty:
    print("Data fetched successfully:")
    print(data.head())  # Print the first few rows of the fetched data

close_prices = data['Close'].squeeze().tolist()

# Debugging output to check data type and content
print("Type of first element in close_prices:", type(close_prices[0]))
print("First few elements in close_prices:", close_prices[:5])

filtered_prices = highpass_filter(close_prices, 14)

if filtered_prices:
    print("Filtered prices:")
    print(filtered_prices[:10])  # Print the first 10 filtered prices

fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=close_prices, mode='lines', name='Original Close Prices'))
fig.add_trace(go.Scatter(x=data.index[2:], y=filtered_prices[2:], mode='lines', name='Highpass Filtered Prices'))

fig.update_layout(title='Comparison of Closing Prices and Highpass Filtered Prices for AAPL',
                  xaxis_title='Date',
                  yaxis_title='Price (USD)',
                  legend_title='Legend')
fig.show()

"""