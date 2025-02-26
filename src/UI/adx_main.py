import streamlit as st
import yfinance as yf
import pandas as pd
import os
import sys
import datetime

# Adjust the system path so that our modules can be imported.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the ADXIndicator class from the Indicators package
from Indicators.adx_indicator import ADXIndicator

st.title("ADX Calculation - Real-Time & Historical Data")

# Input fields for stock symbol
symbol = st.text_input("Enter Stock Symbol:", value="AAPL")

# Date range selection for historical data
start_date = st.date_input("Select Start Date:", value=datetime.date(2021, 1, 1))
end_date = st.date_input("Select End Date:", value=datetime.date(2022, 1, 1))

# Use session state to store fetched data
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None

# Radio button to choose between real-time and historical data
data_type = st.radio("Select Data Type:", ["Historical Data", "Real-Time Data"])

# Function to fetch **historical** data (daily/weekly timeframe)
def fetch_historical_data(symbol, start_date, end_date):
    df = yf.download(symbol, start=start_date, end=end_date, interval="1d")  # Fetching daily historical data
    return df

# Function to fetch **real-time** intraday data (1-minute or 5-minute interval)
def fetch_real_time_data(symbol):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="1d", interval="1m")  # Fetching **real-time** 1-minute intraday data
    return df

# Button to fetch data based on selection
if st.button("Fetch Data"):
    if data_type == "Historical Data":
        df = fetch_historical_data(symbol, start_date, end_date)
        if df.empty:
            st.error("No historical data found for the given symbol and date range.")
        else:
            st.session_state.stock_data = df
            st.write(f"**Historical Stock Data for {symbol}:**")
            st.dataframe(df.tail())
    else:
        df = fetch_real_time_data(symbol)
        if df.empty:
            st.error("No real-time data available for the given symbol.")
        else:
            st.session_state.stock_data = df
            st.write(f"**Real-Time Stock Data for {symbol} (1-min interval):**")
            st.dataframe(df.tail())

# Allow ADX calculation only if stock data is available
if st.session_state.stock_data is not None:
    st.subheader("Customize ADX Parameters")

    # Input field for setting the ADX period
    period = st.number_input("Enter ADX period:", min_value=1, max_value=100, value=14, key="adx_period")

    # Dropdown to select the smoothing method
    smoothing_method = st.selectbox("Select Smoothing Method:", ["SMA", "EMA"])

    # Button to calculate ADX values based on the custom parameters
    if st.button("Calculate ADX"):
        adx_indicator = ADXIndicator(period=period, smoothing_method=smoothing_method)
        df_with_adx = adx_indicator.calculate(st.session_state.stock_data)
        st.write(f"Stock Data with ADX (Period: {period}, Smoothing: {smoothing_method}) for {symbol}:")
        st.dataframe(df_with_adx.tail())
