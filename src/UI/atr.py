import yfinance as yf
import pandas as pd
import streamlit as st
import os
import sys

# Adjust the system path so that our modules can be imported.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def fetch_stock_data(ticker, period='1y', interval='1d'):
    # Fetch historical stock data
    stock_data = yf.download(ticker, period=period, interval=interval)
    return stock_data

def calculate_atr(stock_data, period=14):
    # Calculate the True Range
    stock_data['High-Low'] = stock_data['High'] - stock_data['Low']
    stock_data['High-Close'] = abs(stock_data['High'] - stock_data['Close'].shift(1))
    stock_data['Low-Close'] = abs(stock_data['Low'] - stock_data['Close'].shift(1))

    # True Range is the max of the three
    stock_data['True Range'] = stock_data[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)

    # ATR is the rolling average of the True Range
    stock_data['ATR'] = stock_data['True Range'].rolling(window=period).mean()

    return stock_data

# Streamlit UI
st.title("Stock Data and ATR Calculator")

# Input for stock ticker
ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL)", "AAPL")

# Fetch Stock Data button
if st.button('Fetch Stock Data'):
    if ticker:
        # Fetch the stock data and save it to session_state
        stock_data = fetch_stock_data(ticker)
        st.session_state.stock_data = stock_data  # Save to session_state
        st.write(f"Displaying stock data for {ticker}:")
        st.write(stock_data.head())  # Display the first few rows of stock data
    else:
        st.error("Please enter a valid stock ticker.")

# Calculate ATR button
if st.button('Calculate ATR'):
    if 'stock_data' in st.session_state:
        # Calculate ATR
        stock_data_with_atr = calculate_atr(st.session_state.stock_data)
        
        # Display the stock data along with ATR values
        st.write(f"Displaying Stock Data and Calculated ATR for {ticker}:")
        st.write(stock_data_with_atr[['High', 'Low', 'Close', 'ATR']].tail())  # Display last 5 rows with ATR
    else:
        st.error("Please fetch stock data first.")
