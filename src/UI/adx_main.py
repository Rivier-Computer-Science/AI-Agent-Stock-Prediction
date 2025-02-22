import streamlit as st
import yfinance as yf
import pandas as pd
import os
import sys

# Adjust the system path so that our modules can be imported.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the ADXIndicator class from the Indicators package
from Indicators.adx_indicator import ADXIndicator

st.title("Prototype Trading System - ADX Calculation")

# Input fields for stock symbol and date range
symbol = st.text_input("Enter Stock Symbol:", value="AAPL")
start_date = st.text_input("Enter start date (YYYY-MM-DD):", value="2021-01-01")
end_date = st.text_input("Enter end date (YYYY-MM-DD):", value="2022-01-01")

# Use session state to store the fetched data
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None

# Button to fetch the stock data only
if st.button("Fetch Data"):
    df = yf.download(symbol, start=start_date, end=end_date)
    if df.empty:
        st.error("No data found for the given symbol and date range.")
    else:
        st.session_state.stock_data = df
        st.write(f"Original Stock Data for {symbol}:")
        st.dataframe(df.tail())

# Only allow ADX calculation if the stock data is available
if st.session_state.stock_data is not None:
    # Input field for setting the ADX period
    period = st.number_input("Enter ADX period:", min_value=1, max_value=100, value=14, key="adx_period")

    # Button to calculate ADX values
    if st.button("Calculate ADX"):
        adx_indicator = ADXIndicator(period=period)
        df_with_adx = adx_indicator.calculate(st.session_state.stock_data)
        st.write(f"Stock Data with ADX{period} for {symbol}:")
        st.dataframe(df_with_adx.tail())