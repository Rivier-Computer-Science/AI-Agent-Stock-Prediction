#!/usr/bin/env python3
import os
import pandas as pd
import streamlit as st
from yahooquery import Ticker
from dotenv import load_dotenv

# Load environment variables if needed (e.g., API keys)
load_dotenv()

def fetch_stock_data(ticker_symbol, period='1y'):
    """
    Fetch historical stock data for a given ticker symbol using yahooquery.
    Ensures that the DataFrame contains the required columns: date, high, low, close, and volume.
    """
    st.info(f"Fetching historical data for {ticker_symbol} (period={period})...")
    ticker = Ticker(ticker_symbol)
    data = ticker.history(period=period)
    
    if isinstance(data, pd.DataFrame):
        data.reset_index(inplace=True)
    else:
        st.error("Failed to fetch data as a DataFrame.")
        return None
    
    # Normalize column names to lowercase to avoid capitalization issues
    data.columns = data.columns.str.lower()
    
  
    # Ensure required columns exist
    required_columns = ['date', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        return None
    
    return data

# Function to calculate Volume Price Trend (VPT)
def calculate_vpt(stock_data):
    """
    Calculate the Volume Price Trend (VPT) indicator.
    """
    # Calculate Price Change Percentage
    stock_data['Price Change %'] = stock_data['close'].pct_change()
    
    # Calculate VPT using cumulative sum
    stock_data['VPT'] = (stock_data['volume'] * stock_data['Price Change %']).cumsum()
    
    return stock_data

def main():
    st.title("Volume Price Trend (VPT) Indicator")
    st.write("Calculate the Volume Price Trend (VPT) indicator for your selected stock.")

    # Input for stock symbol and data period.
    ticker_symbol = st.text_input("Enter Stock Symbol:", value="AAPL")
    period_option = st.selectbox("Select Data Period:", options=["1y", "6mo", "3mo", "1mo"], index=0)

    # Button to calculate and display VPT data.
    if st.button("Calculate VPT"):
        # Fetch the historical data.
        data = fetch_stock_data(ticker_symbol, period=period_option)
        if data is not None:
            st.subheader(f"Original Stock Data for {ticker_symbol}")
            st.dataframe(data.tail(10))
            
            # Calculate the VPT values.
            data_with_vpt = calculate_vpt(data)

            st.subheader("Stock Data with Calculated VPT")
            st.dataframe(data_with_vpt[['date', 'close', 'volume', 'VPT']].tail(20))

            # Plot the VPT trend
            st.subheader("VPT Trend Over Time")
            st.line_chart(data_with_vpt.set_index('date')['VPT'])

if __name__ == '__main__':
    main()