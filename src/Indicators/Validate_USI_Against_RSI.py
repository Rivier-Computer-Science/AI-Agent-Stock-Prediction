import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataFetcher:
    def __init__(self, start_date: datetime = None, end_date: datetime = None):
        if start_date is None:
            self.start_date = datetime.today() - timedelta(days=60)
        else:
            self.start_date = start_date

        if end_date is None:
            self.end_date = datetime.today()
        else:
            self.end_date = end_date

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


# Function to calculate SU and SD
def calculate_su_sd(prices):
    su = np.zeros(len(prices))
    sd = np.zeros(len(prices))

    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            su[i] = prices[i] - prices[i - 1]
        elif prices[i] < prices[i - 1]:
            sd[i] = prices[i - 1] - prices[i]
    return su, sd


# Ultimate Smoother Function
def ultimate_smoother(data, period):
    a1 = np.exp(-1.414 * np.pi / period)
    b1 = 2 * a1 * np.cos(1.414 * 180 / period)
    c2 = b1
    c3 = -a1 * a1
    c1 = (1 + c2 - c3) / 4

    smoothed = np.zeros_like(data)
    for i in range(len(data)):
        if i >= 3:
            smoothed[i] = ((1 - c1) * data[i] +
                           (2 * c1 - c2) * data[i - 1] -
                           (c1 + c3) * data[i - 2] +
                           c2 * smoothed[i - 1] +
                           c3 * smoothed[i - 2])
        else:
            smoothed[i] = data[i]
    return smoothed


# USI Calculation
def calculate_usi(su, sd, period):
    smoothed_su = ultimate_smoother(su, period)
    smoothed_sd = ultimate_smoother(sd, period)
    usi = np.zeros(len(su))

    for i in range(len(su)):
        if smoothed_su[i] + smoothed_sd[i] > 0 and smoothed_su[i] > 0.01 and smoothed_sd[i] > 0.01:
            usi[i] = (smoothed_su[i] - smoothed_sd[i]) / (smoothed_su[i] + smoothed_sd[i])
    return usi


# RSI Calculation
def calculate_rsi(prices, period=14):
    su, sd = calculate_su_sd(prices)
    avg_su = np.convolve(su, np.ones(period) / period, mode='same')
    avg_sd = np.convolve(sd, np.ones(period) / period, mode='same')

    rsi = np.zeros(len(prices))
    for i in range(len(prices)):
        if avg_su[i] + avg_sd[i] > 0:
            rsi[i] = avg_su[i] / (avg_su[i] + avg_sd[i])
    return rsi * 100


# Integration Workflow
if __name__ == "__main__":
    # Step 1: Fetch stock data
    data_fetcher = DataFetcher()
    stock_data = data_fetcher.get_stock_data("AAPL")  # Example: Fetch Apple stock data

    # Step 2: Extract closing prices
    closing_prices = stock_data["Close"].values

    # Step 3: Calculate SU and SD
    su, sd = calculate_su_sd(closing_prices)

    # Step 4: Calculate USI
    usi = calculate_usi(su, sd, period=14)

    # Step 5: Calculate RSI
    rsi = calculate_rsi(closing_prices, period=14)

    # Step 6: Compare and display results
    print("USI:", usi)
    print("RSI:", rsi)

