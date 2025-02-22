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
# Helper function to flatten MultiIndex columns (if present)
# -------------------------------------------
def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    If the DataFrame has MultiIndex columns, flatten them.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join([str(i) for i in col]).strip() for col in df.columns.values]
    return df

# -------------------------------------------
# Helper function to standardize column names and remove common trailing tokens
# -------------------------------------------
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert column names to lowercase, remove extra whitespace, and if all columns share a common trailing token,
    remove that trailing token.
    """
    # Convert to lowercase and strip whitespace
    df.columns = df.columns.str.lower().str.strip()
    
    cols = df.columns.tolist()
    # Split each column name by whitespace
    split_cols = [col.split() for col in cols]
    
    # If all columns have at least two tokens
    if all(len(tokens) >= 2 for tokens in split_cols):
        # Extract the last token from each column name
        last_tokens = [tokens[-1] for tokens in split_cols]
        # If all last tokens are the same, remove them from each column name
        if len(set(last_tokens)) == 1:
            new_cols = [' '.join(tokens[:-1]) for tokens in split_cols]
            df.columns = new_cols
    
    return df

# -------------------------------------------
# Function to calculate the Mass Index
# -------------------------------------------
def calculate_mass_index(data: pd.DataFrame, ema_period: int = 9, sum_period: int = 25) -> pd.Series:
    """
    Calculate the Mass Index indicator.
    
    The Mass Index is calculated in the following steps:
      1. Compute the daily range: (high - low)
      2. Calculate the exponential moving average (EMA) of the range using the specified EMA period.
      3. Calculate the EMA of the previously computed EMA.
      4. Compute the ratio: EMA(range) / EMA(EMA(range)).
      5. Calculate the Mass Index as the rolling sum of the ratio over the specified sum period.
    
    Parameters:
    - data: DataFrame containing at least 'high' and 'low' columns.
    - ema_period: Period for calculating the exponential moving averages (default is 9).
    - sum_period: Look-back period over which to sum the ratio (default is 25).
    
    Returns:
    - A Pandas Series representing the Mass Index.
    """
    # Flatten columns (if needed) and standardize the column names
    data = flatten_columns(data)
    data = standardize_columns(data)
    
    # Check if the required columns exist
    required_cols = {"high", "low"}
    if not required_cols.issubset(set(data.columns)):
        st.error(f"Data must contain the columns {required_cols}. Available columns: {list(data.columns)}")
        return pd.Series(dtype=float)
    
    # Calculate the daily price range
    price_range = data['high'] - data['low']
    
    # Calculate the first EMA of the range
    ema_range = price_range.ewm(span=ema_period, adjust=False).mean()
    
    # Calculate the EMA of the previous EMA (Double EMA)
    ema_ema_range = ema_range.ewm(span=ema_period, adjust=False).mean()
    
    # Compute the ratio of the two EMAs
    ratio = ema_range / ema_ema_range
    
    # Calculate the Mass Index as the rolling sum of the ratio over the sum_period
    mass_index = ratio.rolling(window=sum_period).sum()
    
    return mass_index

# -------------------------------------------
# Streamlit UI Code
# -------------------------------------------
def main():
    st.title("Stock Data and Mass Index Calculator")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL")
    period_str = st.sidebar.selectbox("Data Period", options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"], index=3)
    interval = st.sidebar.selectbox("Data Interval", options=["1d", "1wk", "1mo"], index=0)
    
    # Mass Index parameters
    st.sidebar.subheader("Mass Index Parameters")
    ema_period = st.sidebar.number_input("EMA Period", min_value=5, max_value=20, value=9, step=1)
    sum_period = st.sidebar.number_input("Sum Period", min_value=10, max_value=50, value=25, step=1)
    
    # Button to fetch data
    if st.sidebar.button("Fetch Data"):
        st.info(f"Fetching data for {ticker}...")
        stock_data = fetch_stock_data(ticker, period=period_str, interval=interval)
        if not stock_data.empty:
            st.subheader("Fetched Stock Data")
            st.dataframe(stock_data.tail(10))  # Display last 10 rows
            st.session_state['stock_data'] = stock_data
        else:
            st.error("Failed to fetch data. Please check the ticker symbol and parameters.")
    
    # Button to calculate and display Mass Index
    if st.sidebar.button("Calculate Mass Index"):
        if 'stock_data' not in st.session_state:
            st.error("Please fetch the stock data first.")
        else:
            stock_data = st.session_state['stock_data']
            st.info("Calculating Mass Index...")
            mass_index_series = calculate_mass_index(stock_data, ema_period=ema_period, sum_period=sum_period)
            
            # Append the Mass Index to the data for visualization
            stock_data_with_mi = stock_data.copy()
            stock_data_with_mi['Mass Index'] = mass_index_series
            
            st.subheader("Mass Index Chart")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(stock_data_with_mi.index, stock_data_with_mi['Mass Index'], label='Mass Index', color='blue')
            ax.set_title(f"{ticker} Mass Index (EMA Period: {ema_period}, Sum Period: {sum_period})")
            ax.set_xlabel("Date")
            ax.set_ylabel("Mass Index")
            ax.legend()
            st.pyplot(fig)
            
            st.subheader("Data with Mass Index")
            st.dataframe(stock_data_with_mi.tail(10))

if __name__ == "__main__":
    main()
