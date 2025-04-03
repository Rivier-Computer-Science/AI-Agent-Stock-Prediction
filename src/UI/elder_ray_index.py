import pandas as pd
import streamlit as st
from yahooquery import Ticker
from textwrap import dedent
import os
import sys
# CrewAI imports
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# Import the Elder-Ray Analysis Agent from the designated folder
from src.Agents.ElderRay.elder_ray_agent import ElderRayAnalysisAgent


# ---------------------------
# Data Functions
# ---------------------------
def fetch_stock_data(ticker, data_mode, start_date=None, end_date=None, period=None, interval=None):
    """
    Fetch market data using yahooquery.
    For Historical mode: uses start_date and end_date with a forced interval of "1d".
    For Real-Time mode: uses period and interval.
    """
    try:
        t = Ticker(ticker)
        if data_mode == "Real-Time":
            allowed_intervals = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
            if interval not in allowed_intervals:
                st.error(f"Interval '{interval}' is not allowed. Allowed values: {allowed_intervals}")
                return pd.DataFrame()
            data = t.history(period=period, interval=interval)
        else:
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            data = t.history(start=start_str, end=end_str, interval="1d")
        st.write(f"Fetched {len(data)} rows of data.")
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def calculate_moving_average(series, period, ma_type):
    """
    Calculate moving average using either EMA or SMA.
    """
    if ma_type == "EMA":
        return series.ewm(span=period, adjust=False).mean()
    elif ma_type == "SMA":
        return series.rolling(window=period).mean()
    else:
        return series.ewm(span=period, adjust=False).mean()

def calculate_elder_ray_index(stock_data, ma_period=14, ma_type="EMA", price_column="Close"):
    """
    Calculate the Elder-Ray Index (Bull Power and Bear Power).
    Standardizes columns to lowercase, computes the moving average, then calculates:
      Bull Power = high - MA
      Bear Power = low - MA
    """
    # Standardize column names
    stock_data.columns = [col.lower() for col in stock_data.columns]
    price_column = price_column.lower()
    
    stock_data['ma'] = calculate_moving_average(stock_data[price_column], ma_period, ma_type)
    
    high_series = stock_data['high']
    if isinstance(high_series, pd.DataFrame):
        high_series = high_series.iloc[:, 0]
    ma_series = stock_data['ma']
    if isinstance(ma_series, pd.DataFrame):
        ma_series = ma_series.iloc[:, 0]
    low_series = stock_data['low']
    if isinstance(low_series, pd.DataFrame):
        low_series = low_series.iloc[:, 0]
    
    stock_data['bull power'] = high_series - ma_series
    stock_data['bear power'] = low_series - ma_series
    
    return stock_data

def flatten_columns(df):
    """
    Flatten MultiIndex columns and drop extraneous columns (those starting with "index--").
    Reset the DataFrame index afterward.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[[col for col in df.columns if not str(col).startswith("index--")]]
    df.reset_index(drop=True, inplace=True)
    return df

def fetch_current_price(symbol):
    """
    Fetch the current stock price using yahooquery.
    """
    try:
        t = Ticker(symbol)
        price_data = t.price
        if price_data and symbol in price_data and 'regularMarketPrice' in price_data[symbol]:
            return price_data[symbol]['regularMarketPrice']
        else:
            st.error("Failed to fetch current price.")
            return None
    except Exception as e:
        st.error(f"Error fetching current price: {e}")
        return None

# ---------------------------
# Streamlit UI
# ---------------------------
def main():
    st.title("Elder-Ray Index Calculator with Investment Decision Support")
    
    # Common Inputs
    ticker = st.text_input("Enter Stock Symbol (e.g., AAPL, GOOGL):", "AAPL")
    data_mode = st.radio("Select Data Mode", ["Historical", "Real-Time"])
    
    # Moving Average Settings
    ma_period = st.number_input("Enter Moving Average Period", min_value=1, value=14)
    ma_type = st.selectbox("Select Moving Average Type", ["EMA", "SMA"])
    price_column = st.selectbox("Select Price Column for Moving Average", ["Close", "Open", "High", "Low"], index=0)
    
    # Data Mode Specific Inputs
    if data_mode == "Historical":
        start_date = st.date_input("Start Date", pd.to_datetime("2023-01-01"))
        end_date = st.date_input("End Date", pd.to_datetime("2023-12-31"))
        if start_date >= end_date:
            st.error("End Date must be after Start Date.")
            return
        if st.button("Calculate Elder-Ray Index (Historical)"):
            with st.spinner("Fetching historical data and calculating Elder-Ray Index..."):
                stock_data = fetch_stock_data(ticker, data_mode, start_date=start_date, end_date=end_date)
                if stock_data.empty:
                    st.error("No data found for the given stock symbol and date range.")
                    return
                elder_ray_index = calculate_elder_ray_index(stock_data, ma_period, ma_type, price_column)
                elder_ray_index = flatten_columns(elder_ray_index)
                
                st.subheader(f"Elder-Ray Index for {ticker} from {start_date} to {end_date}")
                st.write(elder_ray_index[["bull power", "bear power"]].tail())
                st.line_chart(elder_ray_index[["bull power", "bear power"]])
                
                # Save for CrewAI decision support
                st.session_state.elder_ray_data = elder_ray_index
    else:
        period = st.text_input("Enter Period for Real-Time Data (e.g., 1d, 5d)", "1d")
        interval = st.text_input("Enter Interval (e.g., 1m, 5m, 15m)", "1m")
        if st.button("Calculate Elder-Ray Index (Real-Time)"):
            with st.spinner("Fetching real-time data and calculating Elder-Ray Index..."):
                stock_data = fetch_stock_data(ticker, data_mode, period=period, interval=interval)
                if stock_data.empty:
                    st.error("No data found for the given real-time settings.")
                    return
                elder_ray_index = calculate_elder_ray_index(stock_data, ma_period, ma_type, price_column)
                elder_ray_index = flatten_columns(elder_ray_index)
                
                st.subheader(f"Real-Time Elder-Ray Index for {ticker} (Period: {period}, Interval: {interval})")
                st.write(elder_ray_index[["bull power", "bear power"]].tail())
                st.line_chart(elder_ray_index[["bull power", "bear power"]])
                
                st.session_state.elder_ray_data = elder_ray_index
    
    # ---------------------------
    # CrewAI Investment Decision Support
    # ---------------------------
    if st.button("Get Investment Decision"):
        if "elder_ray_data" not in st.session_state or st.session_state.elder_ray_data.empty:
            st.error("Please calculate the Elder-Ray Index first.")
        else:
            current_price = fetch_current_price(ticker)
            if current_price is None:
                st.error("Unable to fetch current stock price.")
            else:
                # Instantiate the Elder-Ray Analysis Agent (from src/Agents/ElderRay)
                agent_obj = ElderRayAnalysisAgent()
                # Create a task that includes the latest Elder-Ray values and current stock price.
                last_row = st.session_state.elder_ray_data.tail(1).to_string(index=False)
                report = dedent(f"""
                    Elder-Ray Analysis Report:
                    {last_row}
                    Current Stock Price: {current_price}
                    
                    Based on the above Elder-Ray index values and the current stock price, please provide 
                    a clear investment recommendation: BUY, SELL, or HOLD. Include supporting reasoning.
                """)
                decision_task = Task(
                    description=report,
                    agent=agent_obj.elder_ray_investment_advisor(),
                    expected_output="A clear investment recommendation (BUY/SELL/HOLD) with supporting reasoning."
                )
                crew = Crew(agents=[agent_obj.elder_ray_investment_advisor()], tasks=[decision_task], verbose=True)
                decision_result = crew.kickoff()
                st.subheader("Investment Decision")
                st.write(decision_result)

if __name__ == "__main__":
    main()
