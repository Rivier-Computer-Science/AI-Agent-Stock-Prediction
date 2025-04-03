#!/usr/bin/env python3
import os
import sys
import pandas as pd
import streamlit as st
from yahooquery import Ticker
import yfinance as yf
from dotenv import load_dotenv
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# Import the VPT analysis agent from the separate file in src/Agents/VPT
from src.Agents.VPT.vpt_agent import VPTAnalysisAgent

# Load environment variables if needed
load_dotenv()

# ---------------------------
# Data Fetching and VPT Calculation Functions
# ---------------------------
def fetch_historical_data(ticker_symbol, period='1y', start_date=None, end_date=None):
    st.info(f"Fetching historical data for {ticker_symbol}...")
    if start_date and end_date:
        data = yf.download(ticker_symbol, start=start_date, end=end_date)
    else:
        ticker = Ticker(ticker_symbol)
        data = ticker.history(period=period)
    
    if isinstance(data, pd.DataFrame):
        data.reset_index(inplace=True)
    else:
        st.error("Failed to fetch historical data as a DataFrame.")
        return None

    data.columns = data.columns.str.lower()
    required_columns = ['date', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        st.error(f"Missing required columns in historical data: {', '.join(missing_columns)}")
        return None
    return data

def fetch_realtime_data(ticker_symbol, period='1d', interval='1m'):
    st.info(f"Fetching real-time data for {ticker_symbol} (period={period}, interval={interval})...")
    data = yf.download(ticker_symbol, period=period, interval=interval)
    if data.empty:
        st.error("Failed to fetch real-time data.")
        return None
    data.reset_index(inplace=True)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.columns = data.columns.str.lower()
    if 'datetime' in data.columns:
        data.rename(columns={'datetime': 'date'}, inplace=True)
    elif 'date' not in data.columns and data.index.name is not None:
        data.index.name = 'date'
        data.reset_index(inplace=True)
    required_columns = ['date', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        st.error(f"Missing required columns in real-time data: {', '.join(missing_columns)}")
        return None
    return data

def calculate_vpt(stock_data, calc_period=1, weighting_factor=1.0, apply_smoothing=False, smoothing_window=5):
    stock_data['Price Change %'] = stock_data['close'].pct_change(periods=calc_period)
    stock_data['VPT'] = (stock_data['volume'] * stock_data['Price Change %'] * weighting_factor).cumsum()
    if apply_smoothing and smoothing_window > 1:
        stock_data['VPT'] = stock_data['VPT'].rolling(window=int(smoothing_window), min_periods=1).mean()
    return stock_data

def fetch_current_price(symbol: str):
    try:
        ticker = Ticker(symbol)
        price_data = ticker.price
        if price_data and symbol in price_data and 'regularMarketPrice' in price_data[symbol]:
            return price_data[symbol]['regularMarketPrice']
        else:
            st.error("Failed to fetch current price from ticker.price.")
            return None
    except Exception as e:
        st.error(f"Error fetching current price: {e}")
        return None

# ---------------------------
# Streamlit UI
# ---------------------------
def main():
    st.title("Volume Price Trend (VPT) Indicator with Investment Decision Support")
    st.write("Customize, calculate the VPT indicator, and get an investment decision based on VPT and current stock price.")

    ticker_symbol = st.text_input("Enter Stock Symbol:", value="AAPL")
    data_source = st.radio("Select Data Source:", options=["Historical Data", "Real-Time Data"])
    
    custom_date = False
    if data_source == "Historical Data":
        period_option = st.selectbox("Select Historical Data Period:", options=["1y", "6mo", "3mo", "1mo", "Custom"], index=0)
        if period_option == "Custom":
            custom_date = True
            start_date = st.date_input("Start Date")
            end_date = st.date_input("End Date")
    else:
        realtime_period = st.selectbox("Select Real-Time Data Period:", options=["1d", "5d"], index=0)
        realtime_interval = st.selectbox("Select Real-Time Data Interval:", options=["1m", "5m", "15m"], index=0)
    
    st.markdown("---")
    st.subheader("VPT Customization Options")
    calc_period = st.number_input("VPT Calculation Period (in days):", min_value=1, value=1, step=1)
    weighting_factor = st.number_input("Weighting Factor:", min_value=0.0, value=1.0, step=0.1)
    apply_smoothing = st.checkbox("Apply Smoothing to VPT")
    smoothing_window = st.number_input("Smoothing Window (in days):", min_value=1, value=5, step=1) if apply_smoothing else 5
    
    st.markdown("---")
    st.subheader("Chart Customization Options")
    display_chart = st.checkbox("Display VPT Chart", value=True)
    line_color = st.color_picker("Select Line Color", value="#0000FF")
    line_thickness = st.slider("Line Thickness", min_value=1, max_value=10, value=2)
    show_grid = st.checkbox("Show Grid", value=True)
    line_style = st.selectbox("Select Line Style:", options=["solid", "dashed", "dotted", "dashdot"], index=0)
    show_markers = st.checkbox("Show Markers", value=False)
    marker_size = st.slider("Marker Size", min_value=1, max_value=10, value=4) if show_markers else 0
    fig_width = st.slider("Chart Width (inches)", min_value=5, max_value=20, value=10)
    fig_height = st.slider("Chart Height (inches)", min_value=3, max_value=15, value=6)

    if st.button("Calculate VPT"):
        if data_source == "Historical Data":
            if custom_date:
                data = fetch_historical_data(ticker_symbol, start_date=start_date, end_date=end_date)
            else:
                data = fetch_historical_data(ticker_symbol, period=period_option)
        else:
            data = fetch_realtime_data(ticker_symbol, period=realtime_period, interval=realtime_interval)
        
        if data is not None:
            st.subheader(f"Original Stock Data for {ticker_symbol}")
            st.dataframe(data.tail(10))
            
            data_with_vpt = calculate_vpt(
                data,
                calc_period=calc_period,
                weighting_factor=weighting_factor,
                apply_smoothing=apply_smoothing,
                smoothing_window=smoothing_window
            )
            
            st.subheader("Stock Data with Calculated VPT")
            st.dataframe(data_with_vpt[['date', 'close', 'volume', 'VPT']].tail(20))
            
            if display_chart:
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                linestyle_map = {"solid": "-", "dashed": "--", "dotted": ":", "dashdot": "-."}
                ls = linestyle_map.get(line_style, "-")
                if show_markers:
                    ax.plot(data_with_vpt['date'], data_with_vpt['VPT'], color=line_color,
                            linewidth=line_thickness, linestyle=ls, marker='o', markersize=marker_size)
                else:
                    ax.plot(data_with_vpt['date'], data_with_vpt['VPT'], color=line_color,
                            linewidth=line_thickness, linestyle=ls)
                ax.set_title(f"VPT Trend Over Time for {ticker_symbol}")
                ax.set_xlabel("Date")
                ax.set_ylabel("VPT")
                if show_grid:
                    ax.grid(True)
                st.pyplot(fig)
            
            st.session_state.vpt_data = data_with_vpt

    if st.button("Get Investment Decision"):
        if "vpt_data" not in st.session_state or st.session_state.vpt_data.empty:
            st.error("Please calculate VPT values first.")
        else:
            current_price = fetch_current_price(ticker_symbol)
            if current_price is None:
                st.error("Unable to fetch current stock price for decision making.")
            else:
                # Instantiate the VPTAnalysisAgent from the separate file
                vpt_agent_instance = VPTAnalysisAgent()
                advisor_agent = vpt_agent_instance.vpt_trading_advisor()
                decision_task = vpt_agent_instance.vpt_analysis(advisor_agent, st.session_state.vpt_data.copy(), current_price)
                from crewai import Crew
                crew = Crew(agents=[advisor_agent], tasks=[decision_task], verbose=True)
                result = crew.kickoff()
                st.subheader("Investment Decision")
                st.write(result)

if __name__ == '__main__':
    main()
