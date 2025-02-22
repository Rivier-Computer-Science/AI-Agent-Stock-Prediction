import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

class DataFetcher:
    def __init__(self, start_date: datetime = None, end_date: datetime = None):
        """
        Initializes the DataFetcher with a default start date of 60 days ago and an end date of today.
        """
        self.start_date = start_date or datetime.today() - timedelta(days=60)
        self.end_date = end_date or datetime.today()

    def get_stock_data(self, symbol: str, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        """
        Fetches historical stock data for the given symbol.
        """
        start_date = start_date or self.start_date
        end_date = end_date or self.end_date
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        # Download data
        df = yf.download(symbol, start=start_date_str, end=end_date_str)
        df.index = pd.to_datetime(df.index)
        return df

def calculate_su_sd(prices):
    """
    Calculate Strength Up (SU) and Strength Down (SD) from price data.
    """
    su = np.zeros(len(prices))
    sd = np.zeros(len(prices))

    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            su[i] = prices[i] - prices[i - 1]
        elif prices[i] < prices[i - 1]:
            sd[i] = prices[i - 1] - prices[i]

    return su, sd

# Example Usage
if __name__ == "__main__":
    # Initialize the DataFetcher
    data_fetcher = DataFetcher()

    # Fetch historical stock data (e.g., for Apple stock)
    symbol = "AAPL"
    stock_data = data_fetcher.get_stock_data(symbol)

    # Extract closing prices
    closing_prices = stock_data["Close"].values

    # Calculate Strength Up (SU) and Strength Down (SD)
    su, sd = calculate_su_sd(closing_prices)

    # Print results
    print(f"Closing Prices for {symbol}:\n", closing_prices)
    print("Strength Up (SU):", su)
    print("Strength Down (SD):", sd)
