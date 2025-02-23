import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class DataFetcher:
    def __init__(self, start_date: datetime = None, end_date: datetime = None):
        """
        Initializes the DataFetcher with a default start date of 30 days ago and end date of today.

        Args:
            start_date (datetime, optional): The start date for data retrieval. Defaults to 30 days ago.
            end_date (datetime, optional): The end date for data retrieval. Defaults to today.
        """
        if start_date is None:
            self.start_date = datetime.today() - timedelta(days=90)
        else:
            self.start_date = start_date

        if end_date is None:
            self.end_date = datetime.today()
        else:
            self.end_date = end_date

    def get_stock_data(self, symbol: str, start_date: datetime = None, end_date: datetime = None, interval: str="1d") -> pd.DataFrame:
        """
        Fetches historical stock data for the given symbol.

        Args:
            symbol (str): The stock symbol to fetch data for.
            start_date (datetime, optional): The start date for data retrieval. If None, uses self.start_date.
            end_date (datetime, optional): The end date for data retrieval. If None, uses self.end_date.

        Returns:
            pd.DataFrame: A DataFrame containing the historical stock data.

        Raises:
            ValueError: If no data is returned for the symbol.
        """
        print(f"fetching data for ticker {symbol}")
        if start_date is None:
            start_date = self.start_date

        if end_date is None:
            end_date = self.end_date

        # Format dates as strings
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        # Download data from Yahoo Finance
        df = yf.download(symbol, start=start_date_str, end=end_date_str, interval=interval)
        df.columns = df.columns.droplevel('Ticker')  # yfinance changed API's to include the Ticker as a column tuple     

        if df.empty:
            raise ValueError(f"No data returned for symbol {symbol}")

        return df
