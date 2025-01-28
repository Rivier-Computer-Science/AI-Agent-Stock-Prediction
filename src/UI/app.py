import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import streamlit as st
import pandas as pd
from src.Data_Retrieval.data_fetcher import DataFetcher
from src.Indicators.sma import SMAIndicator
from src.Indicators.risk_metrics import RiskMetrics
from src.UI.risk_dashboard import display_risk_metrics
import pandas_ta as ta

st.title("AI Stock Trading System")

page = st.sidebar.selectbox("Choose Analysis", ["Technical Indicators", "Risk Assessment"])


symbol = st.text_input("Enter Stock Symbol:", value="AAPL")


data_fetcher = DataFetcher()
data = data_fetcher.get_stock_data(symbol)

if page == "Technical Indicators":
   
    st.write(f"Original Stock Data for {symbol}:")
    st.dataframe(data.tail())

    if st.button("Calculate SMA"):
        period = st.number_input("Enter SMA period:", min_value=1, max_value=100, value=14)
        sma_indicator = SMAIndicator(period=period)
        data_with_sma = sma_indicator.calculate(data)
        st.write(f"Stock Data with SMA{period} for {symbol}:")
        st.dataframe(data_with_sma.tail())

    if st.button("Calculate RSI"):
        period = st.number_input("Enter RSI period:", min_value=1, max_value=100, value=14)
        data[f"RSI{period}"] = ta.rsi(data['Close'], length=period)
        st.write(f"Stock Data with RSI{period} for {symbol}:")
        st.dataframe(data.tail())

elif page == "Risk Assessment":
    st.subheader(f"Risk Analysis for {symbol}")
    
    risk_calculator = RiskMetrics(data)
    
    # Calculate risk metrics
    risk_data = {
        'returns': data['Close'].pct_change().dropna(),
        'var': risk_calculator.calculate_var(),
        'drawdown': risk_calculator.calculate_drawdown(),
        'volatility': risk_calculator.calculate_volatility(),
        'correlation': risk_calculator.calculate_correlation()
    }
    
    display_risk_metrics(risk_data)

if st.button("Fetch Latest Data"):
    latest_data = data_fetcher.get_stock_data(symbol)
    st.write(f"Latest Stock Data for {symbol}:")
    st.dataframe(latest_data.tail())
