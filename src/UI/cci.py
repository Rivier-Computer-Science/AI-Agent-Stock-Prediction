import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# For real-time auto-refresh (refresh every 60 seconds)
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st.warning("Optional: Install streamlit_autorefresh to enable auto-refresh in real-time mode.")

# -------------------------------------------
# Function to fetch stock data from Yahoo Finance
# -------------------------------------------
def fetch_stock_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch historical (or real-time) stock data for the given ticker symbol.
    
    Parameters:
    - ticker: Stock ticker symbol (e.g., 'AAPL').
    - period: Data period to fetch (e.g., '1y' for one year or '1d' for real-time mode).
    - interval: Data interval (e.g., '1d' for daily data or '1m' for real-time).
    
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
    split_cols = [col.split() for col in cols]
    if all(len(tokens) >= 2 for tokens in split_cols):
        last_tokens = [tokens[-1] for tokens in split_cols]
        if len(set(last_tokens)) == 1:
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
    data = flatten_columns(data)
    data = standardize_columns(data)
    data.columns = data.columns.str.lower().str.strip()
    
    if not {"high", "low", "close"}.issubset(data.columns):
        st.error("Data must contain 'High', 'Low', and 'Close' columns.")
        return pd.Series(dtype=float)
    
    tp = (data['high'] + data['low'] + data['close']) / 3.0
    ma = tp.rolling(window=period).mean()
    md = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
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
    
    # Real-time mode toggle
    realtime_mode = st.sidebar.checkbox("Enable Real-Time Data", value=False)
    
    # Set period and interval based on mode
    if realtime_mode:
        period_str = "1d"
        interval = "1m"
        st.sidebar.info("Real-Time Mode: Using period=1d and interval=1m")
        # Auto-refresh every 60 seconds if st_autorefresh is available
        try:
            st_autorefresh(interval=60000, limit=100, key="datarefresh")
        except Exception as e:
            st.error("Auto-refresh not enabled. Ensure streamlit_autorefresh is installed.")
    else:
        period_str = st.sidebar.selectbox("Data Period", options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"], index=3)
        interval = st.sidebar.selectbox("Data Interval", options=["1d", "1wk", "1mo"], index=0)
    
    cci_period = st.sidebar.number_input("CCI Calculation Period", min_value=5, max_value=50, value=20, step=1)
    
    # Additional Customization Options for Chart Display
    st.sidebar.header("Chart Customization")
    overbought_threshold = st.sidebar.number_input("Overbought Threshold", value=100, step=1)
    oversold_threshold = st.sidebar.number_input("Oversold Threshold", value=-100, step=1)
    show_threshold_lines = st.sidebar.checkbox("Show Threshold Lines", value=True)
    cci_line_color = st.sidebar.text_input("CCI Line Color", value="purple")
    overbought_line_color = st.sidebar.text_input("Overbought Line Color", value="red")
    oversold_line_color = st.sidebar.text_input("Oversold Line Color", value="green")
    show_grid = st.sidebar.checkbox("Show Grid", value=True)
    line_width = st.sidebar.slider("CCI Line Width", min_value=1, max_value=5, value=2)
    chart_width = st.sidebar.number_input("Chart Width", value=10)
    chart_height = st.sidebar.number_input("Chart Height", value=4)
    custom_title = st.sidebar.text_input("Chart Title", value="CCI Chart")
    show_data_table = st.sidebar.checkbox("Show Data Table", value=True)
    
    # Button to fetch data
    if st.sidebar.button("Fetch Data"):
        st.info(f"Fetching data for {ticker}...")
        stock_data = fetch_stock_data(ticker, period=period_str, interval=interval)
        if not stock_data.empty:
            st.subheader("Fetched Stock Data")
            st.dataframe(stock_data.tail(10))
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
            stock_data_with_cci = stock_data.copy()
            stock_data_with_cci['CCI'] = cci_series
            
            st.subheader("CCI Chart")
            fig, ax = plt.subplots(figsize=(chart_width, chart_height))
            ax.plot(stock_data_with_cci.index, stock_data_with_cci['CCI'], label='CCI', color=cci_line_color, linewidth=line_width)
            if show_threshold_lines:
                ax.axhline(overbought_threshold, color=overbought_line_color, linestyle='--', label=f'Overbought ({overbought_threshold})')
                ax.axhline(oversold_threshold, color=oversold_line_color, linestyle='--', label=f'Oversold ({oversold_threshold})')
            ax.set_title(custom_title if custom_title else f"{ticker} CCI (Period: {cci_period})")
            ax.set_xlabel("Date")
            ax.set_ylabel("CCI")
            if show_grid:
                ax.grid(True)
            ax.legend()
            st.pyplot(fig)
            
            if show_data_table:
                st.subheader("Data with CCI")
                st.dataframe(stock_data_with_cci.tail(10))

if __name__ == "__main__":
    main()
