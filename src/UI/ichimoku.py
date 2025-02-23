#!/usr/bin/env python3
import os
import time
import pandas as pd
import streamlit as st
from yahooquery import Ticker
from dotenv import load_dotenv

# For live data auto-refresh (Streamlit version 1.10+ includes st.experimental_rerun, but here we use st_autorefresh)
# If you do not have st_autorefresh built-in, you can install streamlit-autorefresh:
# pip install streamlit-autorefresh
from streamlit_autorefresh import st_autorefresh

# Load environment variables if needed (e.g., API keys)
load_dotenv()

def fetch_stock_data(ticker_symbol, period='1y'):
    """
    Fetch historical stock data for a given ticker symbol using yahooquery.
    Ensures that the DataFrame contains the required columns: date, high, low, and close.
    """
    st.info(f"Fetching historical data for {ticker_symbol} (period={period})...")
    ticker = Ticker(ticker_symbol)
    data = ticker.history(period=period)
    
    if isinstance(data, pd.DataFrame):
        data.reset_index(inplace=True)
    else:
        st.error("Failed to fetch data as a DataFrame.")
        return None
    
    # Ensure required columns exist; rename if necessary.
    for col in ['date', 'high', 'low', 'close']:
        if col not in data.columns:
            if col.capitalize() in data.columns:
                data.rename(columns={col.capitalize(): col}, inplace=True)
            else:
                st.error(f"Required column '{col}' not found in data.")
                return None
    return data

def fetch_realtime_data(ticker_symbol):
    """
    Fetch current market data for a given ticker symbol using yahooquery.
    Returns a DataFrame with the current market data.
    """
    st.info(f"Fetching real-time data for {ticker_symbol}...")
    ticker = Ticker(ticker_symbol)
    try:
        realtime_data = ticker.price
        if realtime_data:
            # Convert the price dictionary to a DataFrame.
            df_rt = pd.DataFrame([realtime_data])
            return df_rt
        else:
            st.error("Failed to fetch real-time data.")
            return None
    except Exception as e:
        st.error(f"Error fetching real-time data: {e}")
        return None

class IchimokuCalculator:
    """
    Calculates Ichimoku Cloud components:
      - Tenkan-sen (Conversion Line)
      - Kijun-sen (Base Line)
      - Senkou Span A (Leading Span A)
      - Senkou Span B (Leading Span B)
      - Chikou Span (Lagging Span)

    Standard parameters (by default):
      - Tenkan-sen period: 9
      - Kijun-sen period: 26
      - Senkou Span B period: 52
      - Displacement: 26

    **Customization Feature:**
      - Smoothing Factor: Applies additional smoothing (via a rolling average) to the computed indicator values.

    Note: The Chikou Span is calculated by shifting the closing price backward by the displacement,
          which is conventional and results in NaN values for the most recent rows.
    """
    def __init__(self, df, tenkan_period=9, kijun_period=26, senkou_b_period=52, displacement=26, smoothing_factor=1):
        self.df = df.copy()
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_b_period = senkou_b_period
        self.displacement = displacement
        self.smoothing_factor = smoothing_factor

    def calculate(self):
        # Ensure the DataFrame is sorted by date.
        if 'date' in self.df.columns:
            self.df.sort_values(by='date', inplace=True)
        
        # Calculate Tenkan-sen: (Highest High + Lowest Low) / 2 over the last `tenkan_period` periods.
        self.df['tenkan_sen'] = (
            self.df['high'].rolling(window=self.tenkan_period, min_periods=self.tenkan_period).max() +
            self.df['low'].rolling(window=self.tenkan_period, min_periods=self.tenkan_period).min()
        ) / 2

        # Calculate Kijun-sen: (Highest High + Lowest Low) / 2 over the last `kijun_period` periods.
        self.df['kijun_sen'] = (
            self.df['high'].rolling(window=self.kijun_period, min_periods=self.kijun_period).max() +
            self.df['low'].rolling(window=self.kijun_period, min_periods=self.kijun_period).min()
        ) / 2

        # Calculate Senkou Span A: (Tenkan-sen + Kijun-sen) / 2, shifted forward by the displacement.
        self.df['senkou_span_a'] = ((self.df['tenkan_sen'] + self.df['kijun_sen']) / 2).shift(self.displacement)

        # Calculate Senkou Span B: (Highest High + Lowest Low) / 2 over the last `senkou_b_period` periods, shifted forward.
        self.df['senkou_span_b'] = (
            self.df['high'].rolling(window=self.senkou_b_period, min_periods=self.senkou_b_period).max() +
            self.df['low'].rolling(window=self.senkou_b_period, min_periods=self.senkou_b_period).min()
        ) / 2
        self.df['senkou_span_b'] = self.df['senkou_span_b'].shift(self.displacement)

        # Calculate Chikou Span: Today's closing price shifted backward by the displacement.
        self.df['chikou_span'] = self.df['close'].shift(-self.displacement)

        # Apply additional smoothing if the smoothing factor is greater than 1.
        if self.smoothing_factor > 1:
            self.df['tenkan_sen'] = self.df['tenkan_sen'].rolling(window=self.smoothing_factor, min_periods=1).mean()
            self.df['kijun_sen'] = self.df['kijun_sen'].rolling(window=self.smoothing_factor, min_periods=1).mean()
            self.df['senkou_span_a'] = self.df['senkou_span_a'].rolling(window=self.smoothing_factor, min_periods=1).mean()
            self.df['senkou_span_b'] = self.df['senkou_span_b'].rolling(window=self.smoothing_factor, min_periods=1).mean()

        return self.df

def main():
    st.title("Ichimoku Cloud Calculation System")
    st.write("Calculate the Ichimoku Cloud indicators for your selected stock with customizable parameters and view real-time data.")

    # Input for stock symbol and data period.
    ticker_symbol = st.text_input("Enter Stock Symbol:", value="AAPL")
    period_option = st.selectbox("Select Data Period:", options=["1y", "6mo", "3mo", "1mo"], index=0)

    st.subheader("Indicator Parameters (Optional)")
    tenkan_period = st.number_input("Tenkan-sen period:", min_value=1, max_value=100, value=9)
    kijun_period = st.number_input("Kijun-sen period:", min_value=1, max_value=100, value=26)
    senkou_b_period = st.number_input("Senkou Span B period:", min_value=1, max_value=200, value=52)
    displacement = st.number_input("Displacement (for Senkou and Chikou):", min_value=1, max_value=100, value=26)
    smoothing_factor = st.number_input("Smoothing Factor:", min_value=1, max_value=10, value=1)

    # Button to calculate and display Ichimoku Cloud data using historical data.
    if st.button("Calculate Ichimoku Cloud"):
        data = fetch_stock_data(ticker_symbol, period=period_option)
        if data is not None:
            st.subheader(f"Original Historical Data for {ticker_symbol}")
            st.dataframe(data.tail(10))
            
            ichimoku_calc = IchimokuCalculator(
                data,
                tenkan_period=tenkan_period,
                kijun_period=kijun_period,
                senkou_b_period=senkou_b_period,
                displacement=displacement,
                smoothing_factor=smoothing_factor
            )
            ichimoku_data = ichimoku_calc.calculate()
            st.subheader("Calculated Ichimoku Cloud Data")
            st.dataframe(ichimoku_data.tail(20))
            st.info("Note: The Chikou Span will show NaN for the most recent rows, which is expected with the conventional calculation.")

    # Button to fetch and display the latest historical data.
    if st.button("Fetch Latest Historical Data"):
        latest_data = fetch_stock_data(ticker_symbol, period=period_option)
        if latest_data is not None:
            st.subheader(f"Latest Historical Data for {ticker_symbol}")
            st.dataframe(latest_data.tail(10))

    # Button to fetch and display real-time market data (one-time fetch).
    if st.button("Fetch Real-Time Data"):
        realtime_data = fetch_realtime_data(ticker_symbol)
        if realtime_data is not None:
            st.subheader(f"Real-Time Data for {ticker_symbol}")
            st.dataframe(realtime_data)
    
    # Button to start a live data feed that updates every 10 seconds.
    if st.button("Start Live Data Feed"):
        # The following autorefresh will cause the entire app to refresh every 10 seconds.
        st_autorefresh(interval=10000, limit=None, key="live_data_autorefresh")
        live_data = fetch_realtime_data(ticker_symbol)
        if live_data is not None:
            st.subheader(f"Live Data Feed for {ticker_symbol}")
            st.dataframe(live_data)

if __name__ == '__main__':
    main()
