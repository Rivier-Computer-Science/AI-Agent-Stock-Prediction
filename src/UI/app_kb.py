import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import streamlit as st
import pandas as pd
from src.Data_Retrieval.data_fetcher import DataFetcher
from src.Indicators.sma import SMAIndicator
from src.Indicators.risk_metrics import RiskMetrics
from src.UI.risk_dashboard import RiskDashboard
import pandas_ta as ta

# Streamlit UI
st.set_page_config(layout="wide")
st.title("AI Stock Trading System")

# Initialize dashboard
risk_dashboard = RiskDashboard()

# Sidebar for navigation
page = st.sidebar.selectbox("Choose Analysis", ["Technical Indicators", "Risk Assessment"])

# Input field to choose stock symbol
symbol = st.text_input("Enter Stock Symbol:", value="AAPL")

# Initialize the DataFetcher and retrieve the data
data_fetcher = DataFetcher()
data = data_fetcher.get_stock_data(symbol)

if page == "Technical Indicators":
    # Display the original data
    st.write(f"Original Stock Data for {symbol}:")
    st.dataframe(data.tail())

    col1, col2 = st.columns(2)
    
    with col1:
        # Add a button to calculate and display SMA
        if st.button("Calculate SMA"):
            period = st.number_input("Enter SMA period:", min_value=1, max_value=100, value=14)
            sma_indicator = SMAIndicator(period=period)
            data_with_sma = sma_indicator.calculate(data)
            st.write(f"Stock Data with SMA{period} for {symbol}:")
            st.dataframe(data_with_sma.tail())

    with col2:
        # Add a button to calculate and display RSI
        if st.button("Calculate RSI"):
            period = st.number_input("Enter RSI period:", min_value=1, max_value=100, value=14)
            data[f"RSI{period}"] = ta.rsi(data['Close'], length=period)
            st.write(f"Stock Data with RSI{period} for {symbol}:")
            st.dataframe(data.tail())

elif page == "Risk Assessment":
    st.subheader(f"Risk Analysis for {symbol}")
    
    # Initialize risk metrics
    risk_calculator = RiskMetrics(data)
    
    # Calculate risk metrics
    risk_data = {
        'returns': data['Close'].pct_change().dropna(),
        'var': risk_calculator.calculate_var(),
        'drawdown': risk_calculator.calculate_drawdown(),
        'volatility': risk_calculator.calculate_volatility(),
        'correlation': risk_calculator.calculate_correlation()
    }
    
    # Display risk dashboard with new interactive features
    risk_dashboard.display_risk_metrics(risk_data)

# Add a button to fetch the latest data
if st.button("Fetch Latest Data"):
    latest_data = data_fetcher.get_stock_data(symbol)
    st.write(f"Latest Stock Data for {symbol}:")
    st.dataframe(latest_data.tail())
