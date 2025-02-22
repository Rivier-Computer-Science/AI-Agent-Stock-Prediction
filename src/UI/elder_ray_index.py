import yfinance as yf
import pandas as pd
import streamlit as st

# Function to fetch stock data from Yahoo Finance
def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch historical stock data from Yahoo Finance.
    
    :param ticker: Stock symbol (e.g., 'AAPL', 'GOOGL')
    :param start_date: Start date in 'YYYY-MM-DD' format
    :param end_date: End date in 'YYYY-MM-DD' format
    :return: DataFrame containing stock data
    """
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    print(stock_data.columns)  # Debug: Check the columns
    return stock_data

# Function to calculate Elder-Ray Index values (Bull Power and Bear Power)
def calculate_elder_ray_index(stock_data, ema_period=14):
    """
    Calculate the Elder-Ray Index values (Bull Power and Bear Power).
    
    :param stock_data: DataFrame containing stock data (should include 'High', 'Low', 'Close')
    :param ema_period: Period for calculating the Exponential Moving Average (default is 14)
    :return: DataFrame with Elder-Ray Index values (Bull Power and Bear Power)
    """
    # Calculate the Exponential Moving Average (EMA)
    stock_data['EMA'] = stock_data['Close'].ewm(span=ema_period, adjust=False).mean()
    
    # Ensure 'High' and 'EMA' are Series (single columns)
    high_series = stock_data['High'].squeeze()  # Convert to Series if it's a DataFrame
    ema_series = stock_data['EMA'].squeeze()    # Convert to Series if it's a DataFrame
    
    # Calculate Bull Power and Bear Power correctly
    stock_data['Bull Power'] = high_series - ema_series
    stock_data['Bear Power'] = stock_data['Low'].squeeze() - ema_series
    
    # Return the full DataFrame with the new columns
    return stock_data

# Streamlit UI
def main():
    # Set up the Streamlit interface
    st.title('Elder-Ray Index Calculator')
    
    # Input fields for Stock Symbol, Start Date, and End Date
    ticker = st.text_input('Enter Stock Symbol (e.g., AAPL, GOOGL):', 'AAPL')
    start_date = st.date_input('Start Date:', pd.to_datetime('2023-01-01'))
    end_date = st.date_input('End Date:', pd.to_datetime('2023-12-31'))
    ema_period = st.number_input('Enter EMA Period (default is 14):', min_value=1, value=14)
    
    # Validate date range
    if start_date >= end_date:
        st.error('End Date must be after Start Date.')
        return
    
    # Button to fetch stock data and calculate Elder-Ray Index
    if st.button('Calculate Elder-Ray Index'):
        with st.spinner('Fetching data and calculating Elder-Ray Index...'):
            # Fetch stock data
            stock_data = fetch_stock_data(ticker, start_date, end_date)
            
            if stock_data.empty:
                st.error('No data found for the given stock symbol and date range. Please check the stock symbol and dates.')
            else:
                # Calculate Elder-Ray Index values
                elder_ray_index = calculate_elder_ray_index(stock_data, ema_period)
                
                # Display the Elder-Ray Index values
                st.subheader(f'Elder-Ray Index for {ticker} ({start_date} to {end_date}):')
                st.write(elder_ray_index[['Bull Power', 'Bear Power']].tail())  # Show the last few rows to validate

if __name__ == "__main__":
    main()