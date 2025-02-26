#!/usr/bin/env python3
import os
import pandas as pd
import streamlit as st
from yahooquery import Ticker
from dotenv import load_dotenv
from datetime import date

# Load environment variables if needed
load_dotenv()

def fetch_stock_data(ticker_symbol, start_date=None, end_date=None):
    """
    Fetch historical stock data for a given ticker symbol using yahooquery.
    If start_date and end_date are provided, data is fetched between those dates.
    Ensures the DataFrame contains the required columns: date, high, low, and close.
    """
    st.info(f"Fetching historical data for {ticker_symbol}...")
    ticker = Ticker(ticker_symbol)
    if start_date and end_date:
        # yahooquery accepts ISO formatted dates.
        data = ticker.history(start=start_date.isoformat(), end=end_date.isoformat())
    else:
        # Fallback if dates are not provided.
        data = ticker.history(period='1y')
    
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

class DonchianCalculator:
    """
    Calculates the Donchian Channels indicator:
      - donchian_high: The highest high over the specified look-back period.
      - donchian_low: The lowest low over the specified look-back period.
    """
    def __init__(self, df, window=20):
        self.df = df.copy()
        self.window = window

    def calculate(self):
        # If a 'date' column exists, sort by it.
        if 'date' in self.df.columns:
            self.df.sort_values(by='date', inplace=True)
        self.df['donchian_high'] = self.df['high'].rolling(window=self.window, min_periods=self.window).max()
        self.df['donchian_low'] = self.df['low'].rolling(window=self.window, min_periods=self.window).min()
        return self.df

def main():
    st.title("Donchian Channels Indicator")
    st.write("Fetch stock data and calculate the Donchian Channels indicator using customizable options.")

    # Sidebar inputs and buttons
    st.sidebar.header("Configuration")
    ticker_symbol = st.sidebar.text_input("Enter Stock Ticker Symbol", value="AAPL")
    
    data_type = st.sidebar.radio("Select Data Type", options=["Historical Data", "Real-Time Data"])
    
    if data_type == "Historical Data":
        st.sidebar.subheader("Historical Data Options")
        start_date = st.sidebar.date_input("Start Date", value=date(2022, 1, 1))
        end_date = st.sidebar.date_input("End Date", value=date.today())
    else:
        st.sidebar.info("Real-Time Data: Recent data with 1-minute intervals will be fetched.")
        start_date = None
        end_date = None

    # Button to fetch stock data
    if st.sidebar.button("Fetch Stock Data"):
        if data_type == "Historical Data":
            data = fetch_stock_data(ticker_symbol, start_date, end_date)
        else:
            data = fetch_realtime_data(ticker_symbol)
        if data is not None:
            st.session_state["data"] = data
            st.success("Stock data fetched successfully!")
        else:
            st.error("Failed to fetch stock data.")

    # Display fetched data if available
    if "data" in st.session_state and st.session_state["data"] is not None:
        st.subheader("Fetched Stock Data")
        st.dataframe(st.session_state["data"].tail(10))

    # Donchian Channel indicator customization options
    st.sidebar.subheader("Donchian Channel Customization")
    donchian_window = st.sidebar.number_input("Donchian Window (Look-back Period)", min_value=1, max_value=100, value=20)

    # Button to calculate the Donchian Channels indicator
    if st.sidebar.button("Calculate Donchian Channels Indicator"):
        if "data" in st.session_state and st.session_state["data"] is not None:
            calc = DonchianCalculator(st.session_state["data"], window=donchian_window)
            data_with_channels = calc.calculate()
            st.session_state["data"] = data_with_channels
            st.success("Donchian Channels calculated successfully!")
        else:
            st.error("No stock data available. Please fetch the stock data first.")

    # Display the calculated data if available
    if "data" in st.session_state and st.session_state["data"] is not None and "donchian_high" in st.session_state["data"].columns:
        st.subheader("Stock Data with Donchian Channels")
        st.dataframe(st.session_state["data"].tail(20))

if __name__ == '__main__':
    main()
