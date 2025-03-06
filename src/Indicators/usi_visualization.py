import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class DataFetcher:
    def __init__(self, start_date: datetime = None, end_date: datetime = None):
        self.start_date = start_date or datetime.today() - timedelta(days=365)  # Default: 1 Year
        self.end_date = end_date or datetime.today()

    def get_stock_data(self, symbol: str) -> pd.DataFrame:
        try:
            df = yf.download(symbol, start=self.start_date.strftime('%Y-%m-%d'), end=self.end_date.strftime('%Y-%m-%d'))
            df.index = pd.to_datetime(df.index)

            if df.empty:
                raise ValueError(f" No data available for {symbol}. Check the stock symbol.")
            
            return df
        except Exception as e:
            print(f" Error fetching data: {e}")
            return pd.DataFrame()  

"""Indicator 1: Ultimate Strength Index (USI)"""
def calculate_su_sd(prices):
    """
    Calculate Strength Up (SU) and Strength Down (SD).
    Fixes the `axis 1 is out of bounds` error by ensuring prices is a 1D array.
    """
    prices = np.asarray(prices).flatten()  
    if len(prices) == 0:
        raise ValueError(" Price data is empty. Ensure stock data is fetched correctly.")

    su = np.maximum(np.diff(prices, prepend=prices[0]), 0)
    sd = np.maximum(-np.diff(prices, prepend=prices[0]), 0)

    return su, sd

def ultimate_smoother(data, period):
    """Smooths data using Ehlers’ Ultimate Smoother."""
    a1 = np.exp(-1.414 * np.pi / period)
    b1 = 2 * a1 * np.cos(1.414 * 180 / period)
    c2, c3 = b1, -a1 * a1
    c1 = (1 + c2 - c3) / 4

    smoothed = np.zeros_like(data)
    for i in range(len(data)):
        smoothed[i] = data[i] if i < 3 else (
            (1 - c1) * data[i] + (2 * c1 - c2) * data[i - 1] - (c1 + c3) * data[i - 2] + c2 * smoothed[i - 1] + c3 * smoothed[i - 2]
        )
    return smoothed

def calculate_usi(prices, period=14):

    """Calculates the Ultimate Strength Index (USI)."""
    su, sd = calculate_su_sd(prices)
    usu, usd = ultimate_smoother(su, period), ultimate_smoother(sd, period)

    usi = np.zeros(len(prices))
    valid_idx = (usu + usd > 0) & (usu > 0.01) & (usd > 0.01)
    usi[valid_idx] = (usu[valid_idx] - usd[valid_idx]) / (usu[valid_idx] + usd[valid_idx])
    
    return usi

""" Indicator 2: Relative Strength Index (RSI)"""

def calculate_rsi(prices, period=14):
    """Calculates RSI using rolling mean."""
    delta = np.diff(prices, prepend=prices[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(window=period, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=1).mean()

    rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))

    return rsi

"""" Indicator 3: MACD (Moving Average Convergence Divergence)"""
def calculate_macd(prices, short_window=12, long_window=26, signal_window=9):
    """Calculates MACD line, Signal line, and MACD histogram."""
    short_ema = pd.Series(prices).ewm(span=short_window, adjust=False).mean()
    long_ema = pd.Series(prices).ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    
    return macd.values, signal.values

""" Visualization Agent: Plot USI, RSI, and MACD"""
class VisualizationAgent:
    @staticmethod
    def plot_indicators(dates, closing_prices, usi_values, rsi_values, macd_values, signal_values):
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            subplot_titles=("Ultimate Strength Index (USI)", "Relative Strength Index (RSI)", "MACD"),
                            vertical_spacing=0.1)

        # 🔹 Plot USI
        fig.add_trace(go.Scatter(x=dates, y=usi_values, mode='lines', name='USI', line=dict(color='blue', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=[0] * len(dates), mode='lines', name='Zero Line', line=dict(color='black', dash='dash')), row=1, col=1)

        # 🔹 Plot RSI
        fig.add_trace(go.Scatter(x=dates, y=rsi_values, mode='lines', name='RSI', line=dict(color='purple', width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=dates, y=[70] * len(dates), mode='lines', name='Overbought (70)', line=dict(color='red', dash='dash')), row=2, col=1)
        fig.add_trace(go.Scatter(x=dates, y=[30] * len(dates), mode='lines', name='Oversold (30)', line=dict(color='green', dash='dash')), row=2, col=1)

        # 🔹 Plot MACD
        fig.add_trace(go.Scatter(x=dates, y=macd_values, mode='lines', name='MACD Line', line=dict(color='orange', width=2)), row=3, col=1)
        fig.add_trace(go.Scatter(x=dates, y=signal_values, mode='lines', name='Signal Line', line=dict(color='red', width=2)), row=3, col=1)
        fig.add_trace(go.Bar(x=dates, y=macd_values - signal_values, name='MACD Histogram', marker_color='grey'), row=3, col=1)

        fig.update_layout(title="Stock Indicators (USI, RSI, MACD)", template="plotly_white", height=800, width=1000)
        fig.show()

if __name__ == "__main__":
    stock_symbol = "AAPL"  # 🔹 Change this to any stock symbol (e.g., "TSLA", "MSFT")

    fetcher = DataFetcher()
    stock_data = fetcher.get_stock_data(stock_symbol)

    if not stock_data.empty:
        closing_prices = stock_data['Close'].dropna().values.flatten()  # Ensure 1D
        dates = stock_data.index

        # Calculate indicators
        usi_values = calculate_usi(closing_prices, period=14)
        rsi_values = calculate_rsi(closing_prices, period=14)
        macd_values, signal_values = calculate_macd(closing_prices)

        """Visualize indicators"""
        
        VisualizationAgent.plot_indicators(dates, closing_prices, usi_values, rsi_values, macd_values, signal_values)
