import yfinance as yf
import pandas as pd
import streamlit as st
import datetime
import os
import sys

# Adjust the system path so that our modules can be imported (if needed).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def fetch_historical_data(ticker, start_date, end_date, interval='1d'):
    """
    Fetch historical stock data for the specified date range and interval.
    """
    stock_data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    return stock_data

def fetch_realtime_data(ticker):
    """
    Fetch near real-time data (past 1 day, 1-minute intervals).
    """
    realtime_data = yf.download(ticker, period='1d', interval='1m')
    return realtime_data

def calculate_atr(stock_data, period=14):
    """
    Calculate the Average True Range (ATR) for the given stock data and period.
    """
    # To avoid SettingWithCopyWarning, work with a copy
    data = stock_data.copy()

    # Calculate the True Range components
    data['High-Low'] = data['High'] - data['Low']
    data['High-Close'] = abs(data['High'] - data['Close'].shift(1))
    data['Low-Close']  = abs(data['Low']  - data['Close'].shift(1))

    # True Range is the max of the three
    data['True Range'] = data[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)

    # ATR is the rolling average of the True Range
    data['ATR'] = data['True Range'].rolling(window=period).mean()

    return data

# Streamlit UI
st.title("Stock Data and ATR Calculator with Real-Time and Historical Integration")

# 1. User input for Stock Ticker
ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL)", "AAPL")

# 2. User-selectable date range for Historical Data
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=datetime.date(2022, 1, 1))
with col2:
    end_date = st.date_input("End Date", value=datetime.date.today())

# 3. Interval selection for Historical Data
interval_options = ["1d", "1wk", "1mo"]  # You can add more if desired (e.g., '1h', '5m', etc.)
selected_interval = st.selectbox("Select Data Interval (Historical)", options=interval_options, index=0)

# 4. ATR Period slider
atr_period = st.slider("ATR Period", min_value=1, max_value=60, value=14, step=1)

# 5. ATR Alert Threshold
atr_threshold = st.number_input(
    "ATR Alert Threshold (optional)",
    min_value=0.0,
    value=0.0,
    step=1.0,
    help="If the ATR exceeds this value, a warning will be displayed."
)

# 6. Columns to Display
available_columns = ["Open", "High", "Low", "Close", "Volume", "ATR"]
selected_columns = st.multiselect(
    "Select Columns to Display",
    options=available_columns,
    default=["High", "Low", "Close", "ATR"]
)

# Section for data fetching
st.subheader("Data Fetching")

# Fetch Historical Data
if st.button('Fetch Historical Data'):
    if ticker:
        historical_data = fetch_historical_data(ticker, start_date, end_date, selected_interval)
        if not historical_data.empty:
            st.session_state.historical_data = historical_data
            st.write(f"Historical data for {ticker} (first 5 rows):")
            st.write(historical_data.head())
        else:
            st.warning("No historical data returned. Please check your date range or ticker symbol.")
    else:
        st.error("Please enter a valid stock ticker.")

# Fetch Real-Time Data
if st.button('Fetch Real-Time Data'):
    if ticker:
        realtime_data = fetch_realtime_data(ticker)
        if not realtime_data.empty:
            st.session_state.realtime_data = realtime_data
            st.write(f"Real-Time data for {ticker} (last 5 rows):")
            st.write(realtime_data.tail())
        else:
            st.warning("No real-time data returned. The provider may not have recent data for this ticker.")
    else:
        st.error("Please enter a valid stock ticker.")

# Section for ATR calculation
st.subheader("ATR Calculation")

# Option to select which dataset to use for ATR
dataset_choice = st.radio(
    "Choose the dataset for ATR calculation:",
    ("Historical Data", "Real-Time Data")
)

if st.button('Calculate ATR'):
    # Determine which dataset to use
    data_key = 'historical_data' if dataset_choice == "Historical Data" else 'realtime_data'
    
    # Check if the chosen dataset is available
    if data_key in st.session_state and not st.session_state[data_key].empty:
        # Calculate ATR
        df_atr = calculate_atr(st.session_state[data_key], period=atr_period)

        # Check if ATR has exceeded the threshold
        latest_atr = df_atr['ATR'].dropna().iloc[-1] if not df_atr['ATR'].dropna().empty else 0.0
        if atr_threshold > 0 and latest_atr > atr_threshold:
            st.warning(f"Alert: The current ATR ({latest_atr:.2f}) exceeds the threshold of {atr_threshold}")

        # Display the last 5 rows of selected columns
        st.write(f"Displaying {dataset_choice} and ATR (period={atr_period}) for {ticker}:")
        columns_to_show = [col for col in selected_columns if col in df_atr.columns]
        if columns_to_show:
            st.write(df_atr[columns_to_show].tail())
        else:
            st.info("No columns selected to display.")

        # Plot ATR if requested
        if "ATR" in columns_to_show:
            st.line_chart(df_atr["ATR"].dropna())
    else:
        st.error(f"No {dataset_choice.lower()} available. Please fetch the data first.")
