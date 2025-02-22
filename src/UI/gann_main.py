import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta

# Import your existing modules
from src.Data_Retrieval.data_fetcher import DataFetcher
from src.Indicators.sma import SMAIndicator  # Import the SMAIndicator class

# ==============================
# Gann Hi-Lo Activator Function
# ==============================
def calculate_gann_hi_lo_activator(df: pd.DataFrame, smoothing_period: int = 0) -> pd.DataFrame:
    """
    Calculates the Gann Hi-Lo Activator indicator.

    For each row:
      - If current close > previous activator:
            activator = min(current low, previous activator)
      - Otherwise:
            activator = max(current high, previous activator)
    
    Optionally applies exponential moving average (EMA) smoothing if smoothing_period > 1.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing 'High', 'Low', and 'Close' columns.
        smoothing_period (int): Period for EMA smoothing. If <= 1, no smoothing is applied.
        
    Returns:
        pd.DataFrame: Original DataFrame with two new columns:
                      - 'Gann Hi Lo': Raw indicator values.
                      - 'Gann Hi Lo Smoothed': Smoothed indicator values.
    """
    # Initialize the activator list with NaN values
    activator = [np.nan] * len(df)
    
    # Set the first value of the activator (commonly, the first low value)
    activator[0] = float(df['Low'].iloc[0])
    
    # Process each row sequentially (simulating real-time processing)
    for i in range(1, len(df)):
        current_close  = float(df['Close'].iloc[i])
        current_low    = float(df['Low'].iloc[i])
        current_high   = float(df['High'].iloc[i])
        prev_activator = float(activator[i - 1])
        
        # Determine the new activator based on directional movement
        if current_close > prev_activator:
            # Uptrend: set activator to the lower of the current low or previous activator
            activator[i] = min(current_low, prev_activator)
        else:
            # Downtrend: set activator to the higher of the current high or previous activator
            activator[i] = max(current_high, prev_activator)
    
    # Add the raw activator values with the desired column name to the DataFrame
    df['Gann Hi Lo'] = activator
    
    # Apply EMA smoothing if requested and ensure the Series has the same index as the DataFrame
    if smoothing_period > 1:
        df['Gann Hi Lo Smoothed'] = pd.Series(activator, index=df.index).ewm(span=smoothing_period, adjust=False).mean()
    else:
        df['Gann Hi Lo Smoothed'] = df['Gann Hi Lo']
    
    return df


# ==============================
# Streamlit UI
# ==============================
st.title("Gann Hi Lo Trading System")

# Input field to choose stock symbol
symbol = st.text_input("Enter Stock Symbol:", value="AAPL")

# Initialize the DataFetcher and retrieve the data
data_fetcher = DataFetcher()
data = data_fetcher.get_stock_data(symbol)

# Display the original stock data
st.write(f"Original Stock Data for {symbol}:")
st.dataframe(data.tail())

# Button and input for calculating the Gann Hi-Lo Activator
gann_smoothing = st.number_input("Enter Gann Hi-Lo Smoothing Period:", min_value=1, max_value=100, value=10, key="gann_smoothing")
if st.button("Calculate Gann Hi-Lo Activator"):
    # Use a copy of the data to avoid modifying the original DataFrame
    data_with_gann = calculate_gann_hi_lo_activator(data.copy(), smoothing_period=gann_smoothing)
    st.write(f"Stock Data with Gann Hi-Lo Activator for {symbol}:")
    st.dataframe(data_with_gann.tail())

# Button to fetch the latest stock data for the selected symbol
if st.button("Fetch Latest Data"):
    latest_data = data_fetcher.get_stock_data(symbol)
    st.write(f"Latest Stock Data for {symbol}:")
    st.dataframe(latest_data.tail())
