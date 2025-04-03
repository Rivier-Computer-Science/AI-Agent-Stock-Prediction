import os
import sys
import logging
from textwrap import dedent
from datetime import datetime
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import streamlit as st
import pandas as pd
import numpy as np
from streamlit_autorefresh import st_autorefresh

# ----- Import Ticker from yahooquery -----
from yahooquery import Ticker

# ----- CrewAI Imports -----
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

# Load environment variables (e.g., API keys)
load_dotenv()

# ----- CrewAI Agent Code for Gann Investment Decision -----
gpt_model = ChatOpenAI(temperature=0.7, model_name="gpt-4o")

class GannAnalysisAgents:
    def gann_investment_advisor(self):
        """
        Returns an agent that provides actionable investment recommendations based on Gann Hi-Lo data.
        """
        return Agent(
            llm=gpt_model,
            role="Gann Investment Advisor",
            goal="Provide actionable investment recommendations based on Gann Hi-Lo Activator data.",
            backstory="You are an experienced technical analyst specializing in Gann indicators. Analyze the latest Gann values and provide clear BUY, SELL, or HOLD recommendations.",
            verbose=True,
            tools=[]
        )
    
    def gann_analysis(self, agent, gann_data, current_price):
        """
        Creates a task for the agent to analyze the latest Gann Hi-Lo data along with the current stock price and provide an investment recommendation.
        """
        latest_raw = gann_data['Gann Hi Lo'].iloc[-1]
        latest_smoothed = gann_data['Gann Hi Lo Smoothed'].iloc[-1]
        description = dedent(f"""
            Analyze the latest Gann Hi-Lo Activator data:
            - Raw Value: {latest_raw}
            - Smoothed Value: {latest_smoothed}
            - Current Stock Price: {current_price}
            
            Based on these values and the current market conditions, provide a detailed investment recommendation.
            Your final answer should clearly indicate whether to BUY, SELL, or HOLD, along with supporting reasoning.
        """)
        return Task(
            description=description,
            agent=agent,
            expected_output=f"Investment decision (BUY/SELL/HOLD) for the stock based on Gann Hi-Lo data and current price."
        )

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
    
    Optionally applies EMA smoothing if smoothing_period > 1.
    
    Returns the DataFrame with two new columns:
      - 'Gann Hi Lo': Raw indicator values.
      - 'Gann Hi Lo Smoothed': Smoothed indicator values.
    """
    # Ensure expected capitalization of columns
    if 'Low' not in df.columns and 'low' in df.columns:
        df.rename(columns={'low': 'Low', 'high': 'High', 'close': 'Close'}, inplace=True)
    
    activator = [np.nan] * len(df)
    activator[0] = float(df['Low'].iloc[0])
    for i in range(1, len(df)):
        current_close  = float(df['Close'].iloc[i])
        current_low    = float(df['Low'].iloc[i])
        current_high   = float(df['High'].iloc[i])
        prev_activator = float(activator[i - 1])
        if current_close > prev_activator:
            activator[i] = min(current_low, prev_activator)
        else:
            activator[i] = max(current_high, prev_activator)
    df['Gann Hi Lo'] = activator
    if smoothing_period > 1:
        df['Gann Hi Lo Smoothed'] = pd.Series(activator, index=df.index)\
                                         .ewm(span=smoothing_period, adjust=False).mean()
    else:
        df['Gann Hi Lo Smoothed'] = df['Gann Hi Lo']
    return df

# ==============================
# Data Retrieval Functions using yahooquery
# ==============================
def fetch_historical_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical market data for the given symbol between specified dates using yahooquery.
    """
    try:
        ticker = Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        if isinstance(data, pd.DataFrame):
            data.reset_index(inplace=True)
            data['date'] = pd.to_datetime(data['date'], utc=True).dt.tz_convert(None)
            for col in ['date', 'high', 'low', 'close']:
                if col not in data.columns and col.capitalize() in data.columns:
                    data.rename(columns={col.capitalize(): col}, inplace=True)
            return data
        else:
            st.error("Failed to fetch historical data as a DataFrame.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching historical data: {e}")
        return pd.DataFrame()

def fetch_realtime_data(symbol: str) -> pd.DataFrame:
    """
    Fetch real-time intraday market data for the given symbol using yahooquery.
    Real-time data is fetched as today's data with a 1-minute interval.
    """
    try:
        ticker = Ticker(symbol)
        data = ticker.history(period="1d", interval="1m")
        if isinstance(data, pd.DataFrame):
            data.reset_index(inplace=True)
            return data
        else:
            st.error("Failed to fetch real-time data as a DataFrame.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching real-time data: {e}")
        return pd.DataFrame()

def fetch_current_price(symbol: str):
    """
    Fetch the current stock price for the given symbol using yahooquery's ticker.price.
    """
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

# ==============================
# Streamlit UI
# ==============================
st.title("Gann Hi Lo Trading System")

# Initialize session state for storing fetched data and calculated Gann values
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame()
if "gann_data" not in st.session_state:
    st.session_state.gann_data = pd.DataFrame()

# --- Stock Symbol Input ---
symbol = st.text_input("Enter Stock Symbol:", value="AAPL")

# --- Data Mode Selection ---
data_mode = st.selectbox("Select Data Mode:", options=["Historical", "Real-Time"])

# ==============================
# Historical Data Section
# ==============================
if data_mode == "Historical":
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=pd.to_datetime("2023-01-01"), key="start_date")
    with col2:
        end_date = st.date_input("End Date", value=pd.to_datetime("today"), key="end_date")
    
    if st.button("Fetch Historical Data"):
        st.session_state.data = fetch_historical_data(
            symbol, 
            start_date.strftime("%Y-%m-%d"), 
            end_date.strftime("%Y-%m-%d")
        )
        if not st.session_state.data.empty:
            st.write(f"Historical Stock Data for {symbol}:")
            st.dataframe(st.session_state.data.tail())
        else:
            st.error("No data found for the selected date range.")

    if not st.session_state.data.empty:
        st.subheader("Gann Hi-Lo Customization Settings (Historical Data)")
        gann_smoothing = st.number_input("Enter Gann Hi-Lo Smoothing Period:", min_value=1, max_value=100, value=10, key="gann_smoothing_hist")
        if st.button("Calculate Gann Hi-Lo Activator (Historical)"):
            data_with_gann = calculate_gann_hi_lo_activator(st.session_state.data.copy(), smoothing_period=gann_smoothing)
            st.write(f"Historical Stock Data with Gann Hi-Lo Activator for {symbol}:")
            st.dataframe(data_with_gann.tail())
            st.session_state.gann_data = data_with_gann

# ==============================
# Real-Time Data Section
# ==============================
else:
    auto_refresh = st.checkbox("Auto Refresh Real-Time Data Every 5 Seconds", value=False)
    if auto_refresh:
        st_autorefresh(interval=5000, limit=0, key="real_time_autorefresh")

    if st.button("Fetch Real-Time Data"):
        current_price = fetch_current_price(symbol)
        if current_price is not None:
            st.write(f"Current Stock Price for {symbol}: {current_price}")
        st.session_state.data = fetch_realtime_data(symbol)
        if not st.session_state.data.empty:
            st.write("Real-Time Intraday Data:")
            st.dataframe(st.session_state.data.tail())
        else:
            st.error("No real-time data found.")

    if not st.session_state.data.empty:
        st.subheader("Gann Hi-Lo Customization Settings (Real-Time Data)")
        gann_smoothing_rt = st.number_input("Enter Gann Hi-Lo Smoothing Period:", min_value=1, max_value=100, value=10, key="gann_smoothing_rt")
        if st.button("Calculate Gann Hi-Lo Activator (Real-Time)"):
            data_with_gann = calculate_gann_hi_lo_activator(st.session_state.data.copy(), smoothing_period=gann_smoothing_rt)
            st.write(f"Real-Time Intraday Data with Gann Hi-Lo Activator for {symbol}:")
            st.dataframe(data_with_gann.tail())
            st.session_state.gann_data = data_with_gann

# ==============================
# CrewAI Investment Decision Support
# ==============================
st.subheader("CrewAI Investment Decision Support")

if st.button("Get Investment Decision"):
    if st.session_state.gann_data.empty:
        st.error("Please calculate Gann Hi-Lo Activator values first.")
    else:
        current_price = fetch_current_price(symbol)
        if current_price is None:
            st.error("Unable to fetch current stock price for decision making.")
        else:
            agents = GannAnalysisAgents()
            advisor_agent = agents.gann_investment_advisor()
            decision_task = agents.gann_analysis(advisor_agent, st.session_state.gann_data.copy(), current_price)
            crew = Crew(agents=[advisor_agent], tasks=[decision_task], verbose=True)
            result = crew.kickoff()
            st.subheader("Investment Decision")
            st.write(result)

if __name__ == '__main__':
    print()
