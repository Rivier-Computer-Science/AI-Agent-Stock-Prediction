import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import streamlit as st
import pandas as pd
import datetime
import yfinance as yf

# Import your existing modules
from src.Data_Retrieval.data_fetcher import DataFetcher
from src.Indicators.trix import calculate_trix

# -----------------------------------
# Streamlit App: TRIX Indicator
# -----------------------------------

st.title("TRIX Indicator: Customization & Data Integration")

# 1. User Input for Symbol
symbol = st.text_input("Enter Stock Symbol:", value="AAPL")

# Get the current stock price using yfinance
try:
    ticker = yf.Ticker(symbol)
    current_price = ticker.info.get("regularMarketPrice")
    if current_price is not None:
        st.write(f"Current Stock Price for {symbol}: ${current_price:.2f}")
    else:
        st.write("Current stock price not available.")
except Exception as e:
    st.error(f"Error retrieving current stock price: {e}")

# 2. Choose Data Source (Real-Time vs Historical)
data_source_choice = st.radio(
    "Select Data Source:",
    options=["Real-Time", "Historical"],
    index=0
)

# 3. (Optional) Date Range Selection for Historical Data
#    Only show date inputs if the user chooses "Historical"
start_date = None
end_date = None

if data_source_choice == "Historical":
    st.write("Select the date range for historical data:")
    today = datetime.date.today()
    default_start = today - datetime.timedelta(days=30)  # Example: last 30 days
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=default_start)
    with col2:
        end_date = st.date_input("End Date", value=today)

    # Basic validation check
    if start_date > end_date:
        st.warning("Start date cannot be after end date. Please adjust your selection.")

# 4. Fetch Stock Data
data_fetcher = DataFetcher()

if data_source_choice == "Real-Time":
    # Fetch intraday data for today using yfinance.
    # Note: For 1-minute intervals, yfinance may return data for multiple days.
    # So, we filter to keep only today's data.
    data = ticker.history(period="1d", interval="1m")
    today_date = datetime.date.today()
    # Filter the DataFrame to include only rows from today.
    data = data[data.index.date == today_date]
else:
    # Historical data fetch with date range using the existing DataFetcher
    data = data_fetcher.get_stock_data(symbol, start_date=start_date, end_date=end_date)

st.write(f"Showing data for: {symbol}")
st.dataframe(data.tail())

# 5. Inputs for TRIX Calculation (Customization Features)
st.subheader("TRIX Indicator Parameters")
trix_length = st.number_input("TRIX Length (EMA periods):", min_value=1, max_value=100, value=14, key="trix_length")
trix_signal = st.number_input("TRIX Signal Period:", min_value=1, max_value=100, value=9, key="trix_signal")

# Optional: Additional smoothing or advanced parameter
apply_additional_smoothing = st.checkbox("Apply Additional Smoothing?")  # Example toggle

# 6. Calculate and Display TRIX
if st.button("Calculate TRIX"):
    # Copy data to avoid modifying the original DataFrame
    data_copy = data.copy()
    # Compute TRIX with user-defined parameters
    data_with_trix = calculate_trix(data_copy, length=trix_length, signal=trix_signal)
    
    # Optional: If the user wants extra smoothing or custom logic
    if apply_additional_smoothing:
        # Example: Smooth the TRIX output with a short EMA
        data_with_trix['TRIX_SMOOTHED'] = data_with_trix['TRIX'].ewm(span=5, adjust=False).mean()

    st.write(f"TRIX Calculation Results for {symbol}:")
    st.dataframe(data_with_trix.tail())

    # Validate that the TRIX values are being computed
    if data_with_trix['TRIX'].isna().all():
        st.warning("All TRIX values are NaN. Check if there's enough historical data or adjust the date range.")
    else:
        st.success("TRIX calculation complete. Parameters successfully applied.")
