import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# -------------------------------------------
# Function to fetch stock data from Yahoo Finance
# -------------------------------------------
def fetch_stock_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch historical stock data for the given ticker symbol.
    
    Parameters:
    - ticker: Stock ticker symbol (e.g., 'AAPL').
    - period: Data period to fetch (e.g., '1y' for one year).
    - interval: Data interval (e.g., '1d' for daily data).
    
    Returns:
    - DataFrame containing the stock data.
    """
    try:
        data = yf.download(ticker, period=period, interval=interval)
        if data.empty:
            st.error("No data found. Please check the ticker symbol.")
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# -------------------------------------------
# Helper function to flatten MultiIndex columns
# -------------------------------------------
def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    If the DataFrame has MultiIndex columns, flatten them.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join([str(i) for i in col]).strip() for col in df.columns.values]
    return df

# -------------------------------------------
# Helper function to standardize column names
# -------------------------------------------
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove common trailing tokens from the column names if they are present.
    For example, if all columns end with the same ticker name, remove it.
    """
    cols = df.columns.tolist()
    # Split each column name by whitespace
    split_cols = [col.split() for col in cols]
    # Check if all columns have at least two tokens and share the same last token
    if all(len(tokens) >= 2 for tokens in split_cols):
        last_tokens = [tokens[-1] for tokens in split_cols]
        if len(set(last_tokens)) == 1:
            # Remove the trailing token from each column name
            new_cols = [' '.join(tokens[:-1]) for tokens in split_cols]
            df.columns = new_cols
    return df

# -------------------------------------------
# Function to calculate the Commodity Channel Index (CCI)
# -------------------------------------------
def calculate_cci(data: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Calculate the Commodity Channel Index (CCI) for a DataFrame with columns: High, Low, Close.
    
    Parameters:
    - data: DataFrame containing at least 'High', 'Low', 'Close' columns.
    - period: Look-back period for the CCI calculation.
    
    Returns:
    - A Pandas Series representing the CCI.
    """
    # Flatten columns if necessary
    data = flatten_columns(data)
    
    # Standardize the column names by removing trailing tokens (e.g., ticker name)
    data = standardize_columns(data)
    
    # Convert column names to lowercase for consistency
    data.columns = data.columns.str.lower().str.strip()
    
   
    # Check if required columns exist
    if not {"high", "low", "close"}.issubset(data.columns):
        st.error("Data must contain 'High', 'Low', and 'Close' columns.")
        return pd.Series(dtype=float)
    
    # Calculate the Typical Price
    tp = (data['high'] + data['low'] + data['close']) / 3.0

    # Calculate the moving average of the Typical Price
    ma = tp.rolling(window=period).mean()
    
    # Calculate the mean deviation
    md = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    
    # Calculate CCI using the formula: (TP - MA) / (0.015 * MD)
    cci = (tp - ma) / (0.015 * md)
    
    return cci

# -------------------------------------------
# Streamlit UI Code
# -------------------------------------------
def main():
    st.title("Stock Data and CCI Calculator")
    
    # Sidebar for user inputs
    st.sidebar.header("Configuration")
    ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL")
    period_str = st.sidebar.selectbox("Data Period", options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"], index=3)
    interval = st.sidebar.selectbox("Data Interval", options=["1d", "1wk", "1mo"], index=0)
    cci_period = st.sidebar.number_input("CCI Calculation Period", min_value=5, max_value=50, value=20, step=1)
    
    # Button to fetch data
    if st.sidebar.button("Fetch Data"):
        st.info(f"Fetching data for {ticker}...")
        stock_data = fetch_stock_data(ticker, period=period_str, interval=interval)
        if not stock_data.empty:
            st.subheader("Fetched Stock Data")
            st.dataframe(stock_data.tail(10))  # Display last 10 rows
            st.session_state['stock_data'] = stock_data
        else:
            st.error("Failed to fetch data.")
    
    # Button to calculate and display CCI
    if st.sidebar.button("Calculate CCI"):
        if 'stock_data' not in st.session_state:
            st.error("Please fetch the stock data first.")
        else:
            stock_data = st.session_state['stock_data']
            st.info("Calculating CCI...")
            cci_series = calculate_cci(stock_data, period=cci_period)
            
            # Add CCI to the DataFrame
            stock_data_with_cci = stock_data.copy()
            stock_data_with_cci['CCI'] = cci_series
            
            st.subheader("CCI Chart")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(stock_data_with_cci.index, stock_data_with_cci['CCI'], label='CCI', color='purple')
            ax.axhline(100, color='red', linestyle='--', label='Overbought (+100)')
            ax.axhline(-100, color='green', linestyle='--', label='Oversold (-100)')
            ax.set_title(f"{ticker} CCI (Period: {cci_period})")
            ax.set_xlabel("Date")
            ax.set_ylabel("CCI")
            ax.legend()
            st.pyplot(fig)
            
            st.subheader("Data with CCI")
            st.dataframe(stock_data_with_cci.tail(10))

if __name__ == "__main__":
    main()
