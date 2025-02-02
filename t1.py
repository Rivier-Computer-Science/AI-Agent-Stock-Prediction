import yfinance as yf
import pandas as pd

# Function to fetch real-time stock data for multiple tickers
def fetch_data(tickers, start_date, end_date):
    """
    Fetches real-time stock data from Yahoo Finance
    :param tickers: List of stock symbols (e.g., ['AAPL', 'GOOG', 'TSLA'])
    :param start_date: The start date for data retrieval (e.g., '2022-01-01')
    :param end_date: The end date for data retrieval (e.g., '2023-01-01')
    :return: DataFrame with stock data
    """
    data = yf.download(tickers, start=start_date, end=end_date)
    return data

# Sample usage: Fetch data for 10 stocks
tickers = ['AAPL', 'GOOG', 'TSLA', 'AMZN', 'MSFT', 'NFLX', 'META', 'NVDA', 'BABA', 'INTC']
start_date = '2023-01-01'
end_date = '2023-12-31'

stock_data = fetch_data(tickers, start_date, end_date)

# Display the fetched data
print(stock_data.head())
