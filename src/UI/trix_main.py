import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import streamlit as st
import pandas as pd

# Import your existing modules
#from src.Data_Retrieval.data_fetcher import DataFetcher
#from src.Indicators.trix import calculate_trix

from src.Data_Retrieval.data_fetcher import DataFetcher
from src.Indicators.trix import calculate_trix

st.title("TRIX Indicator Calculation")

# --- 1. User Input for Symbol ---
symbol = st.text_input("Enter Stock Symbol:", value="AAPL")

# --- 2. Fetch Stock Data ---
data_fetcher = DataFetcher()
data = data_fetcher.get_stock_data(symbol)

# Display the fetched stock data
st.write(f"Fetched Stock Data for {symbol}:")
st.dataframe(data.tail())

# --- 3. Inputs for TRIX Calculation ---
trix_length = st.number_input("Enter TRIX Length:", min_value=1, max_value=100, value=14, key="trix_length")
trix_signal = st.number_input("Enter TRIX Signal Period:", min_value=1, max_value=100, value=9, key="trix_signal")

# --- 4. Calculate TRIX and Display ---
if st.button("Calculate TRIX"):
    # Copy data to avoid modifying the original DataFrame
    data_with_trix = calculate_trix(data.copy(), length=trix_length, signal=trix_signal)
    st.write(f"TRIX Calculation Results for {symbol}:")
    st.dataframe(data_with_trix.tail())

