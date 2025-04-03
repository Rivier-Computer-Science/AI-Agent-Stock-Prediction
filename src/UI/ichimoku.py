#!/usr/bin/env python3
import os
import sys
import time
import pandas as pd
import streamlit as st
from yahooquery import Ticker
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh

# CrewAI imports
from crewai import Agent, Task, Crew
from textwrap import dedent
from langchain_openai import ChatOpenAI


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import tools for CrewAI (ensure these modules exist in your project)
from src.Agents.Analysis.Tools.browser_tools import BrowserTools
from src.Agents.Analysis.Tools.calculator_tools import CalculatorTools
from src.Agents.Analysis.Tools.search_tools import SearchTools
from src.Agents.Analysis.Tools.sec_tools import SECTools
from langchain_community.tools import YahooFinanceNewsTool

# Load environment variables (e.g., API keys)
load_dotenv()

# ----- CrewAI Agent Code for Ichimoku Cloud Investment Decision -----
gpt_model = ChatOpenAI(
    temperature=0.7,
    model_name="gpt-4o"
)

class IchimokuAnalysisAgents:
    def ichimoku_cloud_investment_advisor(self):
        """
        Returns an agent that analyzes Ichimoku Cloud data to provide actionable investment advice.
        """
        return Agent(
            llm=gpt_model,
            role="Ichimoku Cloud Investment Advisor",
            goal="Provide actionable investment recommendations based on Ichimoku Cloud indicator data.",
            backstory="You are an experienced technical analyst specializing in the Ichimoku Cloud system. Analyze the latest indicator values and provide clear buy, sell, or hold signals.",
            verbose=True,
            tools=[
                BrowserTools.scrape_and_summarize_website,
                SearchTools.search_internet,
                CalculatorTools.calculate,
                SECTools.search_10q,
                SECTools.search_10k,
                YahooFinanceNewsTool()
            ]
        )

    def ichimoku_cloud_analysis(self, agent, ichimoku_data):
        """
        Creates a task for the agent to analyze the latest Ichimoku Cloud indicator data and provide a recommendation.
        """
        latest_tenkan = ichimoku_data['tenkan_sen'].iloc[-1]
        latest_kijun = ichimoku_data['kijun_sen'].iloc[-1]
        latest_senkou_a = ichimoku_data['senkou_span_a'].iloc[-1]
        latest_senkou_b = ichimoku_data['senkou_span_b'].iloc[-1]
        latest_chikou = ichimoku_data['chikou_span'].iloc[-1]

        description = dedent(f"""
            Analyze the latest Ichimoku Cloud indicator data:
            - Tenkan-sen (Conversion Line): {latest_tenkan}
            - Kijun-sen (Base Line): {latest_kijun}
            - Senkou Span A (Leading Span A): {latest_senkou_a}
            - Senkou Span B (Leading Span B): {latest_senkou_b}
            - Chikou Span (Lagging Span): {latest_chikou}

            Based on these values and the current market conditions, provide a detailed investment recommendation.
            Your final answer should clearly indicate whether to BUY, SELL, or HOLD, along with supporting reasoning.
        """)

        return Task(
            description=description,
            agent=agent,
            expected_output="A detailed analysis and a clear buy, sell, or hold recommendation based on the Ichimoku Cloud data."
        )

# ----- Data Fetching Functions -----
def fetch_stock_data(ticker_symbol, period='1y'):
    """
    Fetch historical stock data for a given ticker symbol using yahooquery.
    Ensures that the DataFrame contains the required columns: date, high, low, and close.
    """
    st.info(f"Fetching historical data for {ticker_symbol} (period={period})...")
    ticker = Ticker(ticker_symbol)
    data = ticker.history(period=period)
    
    if isinstance(data, pd.DataFrame):
        data.reset_index(inplace=True)
        # Convert the 'date' column to datetime in UTC then remove the timezone to make it naive.
        data['date'] = pd.to_datetime(data['date'], utc=True).dt.tz_convert(None)
    else:
        st.error("Failed to fetch data as a DataFrame.")
        return None
    
    for col in ['date', 'high', 'low', 'close']:
        if col not in data.columns:
            if col.capitalize() in data.columns:
                data.rename(columns={col.capitalize(): col}, inplace=True)
            else:
                st.error(f"Required column '{col}' not found in data.")
                return None
    return data


def fetch_realtime_data(ticker_symbol):
    """
    Fetch current market data for a given ticker symbol using yahooquery.
    Returns a DataFrame with the current market data.
    """
    st.info(f"Fetching real-time data for {ticker_symbol}...")
    ticker = Ticker(ticker_symbol)
    try:
        realtime_data = ticker.price
        if realtime_data:
            df_rt = pd.DataFrame([realtime_data])
            return df_rt
        else:
            st.error("Failed to fetch real-time data.")
            return None
    except Exception as e:
        st.error(f"Error fetching real-time data: {e}")
        return None

# ----- Ichimoku Cloud Calculation -----
class IchimokuCalculator:
    def __init__(self, df, tenkan_period=9, kijun_period=26, senkou_b_period=52, displacement=26, smoothing_factor=1):
        self.df = df.copy()
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_b_period = senkou_b_period
        self.displacement = displacement
        self.smoothing_factor = smoothing_factor

    def calculate(self):
        if 'date' in self.df.columns:
            self.df.sort_values(by='date', inplace=True)
        
        self.df['tenkan_sen'] = (
            self.df['high'].rolling(window=self.tenkan_period, min_periods=self.tenkan_period).max() +
            self.df['low'].rolling(window=self.tenkan_period, min_periods=self.tenkan_period).min()
        ) / 2

        self.df['kijun_sen'] = (
            self.df['high'].rolling(window=self.kijun_period, min_periods=self.kijun_period).max() +
            self.df['low'].rolling(window=self.kijun_period, min_periods=self.kijun_period).min()
        ) / 2

        self.df['senkou_span_a'] = ((self.df['tenkan_sen'] + self.df['kijun_sen']) / 2).shift(self.displacement)

        self.df['senkou_span_b'] = (
            self.df['high'].rolling(window=self.senkou_b_period, min_periods=self.senkou_b_period).max() +
            self.df['low'].rolling(window=self.senkou_b_period, min_periods=self.senkou_b_period).min()
        ) / 2
        self.df['senkou_span_b'] = self.df['senkou_span_b'].shift(self.displacement)

        self.df['chikou_span'] = self.df['close'].shift(-self.displacement)

        if self.smoothing_factor > 1:
            self.df['tenkan_sen'] = self.df['tenkan_sen'].rolling(window=self.smoothing_factor, min_periods=1).mean()
            self.df['kijun_sen'] = self.df['kijun_sen'].rolling(window=self.smoothing_factor, min_periods=1).mean()
            self.df['senkou_span_a'] = self.df['senkou_span_a'].rolling(window=self.smoothing_factor, min_periods=1).mean()
            self.df['senkou_span_b'] = self.df['senkou_span_b'].rolling(window=self.smoothing_factor, min_periods=1).mean()

        return self.df

# ----- Main Streamlit Application -----
def main():
    st.title("Ichimoku Cloud Calculation System")
    st.write("Calculate Ichimoku Cloud indicators for your selected stock with customizable parameters and view real-time data.")
    
    # Input Section
    ticker_symbol = st.text_input("Enter Stock Symbol:", value="AAPL")
    period_option = st.selectbox("Select Data Period:", options=["1y", "6mo", "3mo", "1mo"], index=0)
    st.subheader("Indicator Parameters (Optional)")
    tenkan_period = st.number_input("Tenkan-sen period:", min_value=1, max_value=100, value=9)
    kijun_period = st.number_input("Kijun-sen period:", min_value=1, max_value=100, value=26)
    senkou_b_period = st.number_input("Senkou Span B period:", min_value=1, max_value=200, value=52)
    displacement = st.number_input("Displacement (for Senkou and Chikou):", min_value=1, max_value=100, value=26)
    smoothing_factor = st.number_input("Smoothing Factor:", min_value=1, max_value=10, value=1)


    # Button to Fetch Latest Historical Data
    if st.button("Fetch Latest Historical Data"):
        latest_data = fetch_stock_data(ticker_symbol, period=period_option)
        if latest_data is not None:
            st.subheader(f"Latest Historical Data for {ticker_symbol}")
            st.dataframe(latest_data.tail(10))
    
    # Button to Fetch Real-Time Data
    if st.button("Fetch Real-Time Data"):
        realtime_data = fetch_realtime_data(ticker_symbol)
        if realtime_data is not None:
            st.subheader(f"Real-Time Data for {ticker_symbol}")
            st.dataframe(realtime_data)
    
    # Button to Calculate Ichimoku Cloud
    if st.button("Calculate Ichimoku Cloud"):
        data = fetch_stock_data(ticker_symbol, period=period_option)
        if data is not None:
            st.subheader(f"Original Historical Data for {ticker_symbol}")
            st.dataframe(data.tail(10))
            ichimoku_calc = IchimokuCalculator(
                data,
                tenkan_period=tenkan_period,
                kijun_period=kijun_period,
                senkou_b_period=senkou_b_period,
                displacement=displacement,
                smoothing_factor=smoothing_factor
            )
            ichimoku_data = ichimoku_calc.calculate()
            st.subheader("Calculated Ichimoku Cloud Data")
            st.dataframe(ichimoku_data.tail(20))
            st.info("Note: The Chikou Span will show NaN for the most recent rows, which is expected with the conventional calculation.")
            st.session_state.ichimoku_data = ichimoku_data
    
    # Button to Get Investment Decision using CrewAI
    if st.button("Get Investment Decision"):
        if "ichimoku_data" not in st.session_state:
            st.error("Please calculate the Ichimoku Cloud data first.")
        else:
            ichimoku_data = st.session_state.ichimoku_data
            agents = IchimokuAnalysisAgents()
            advisor_agent = agents.ichimoku_cloud_investment_advisor()
            analysis_task = agents.ichimoku_cloud_analysis(advisor_agent, ichimoku_data)
            crew = Crew(
                agents=[advisor_agent],
                tasks=[analysis_task],
                verbose=True
            )
            result = crew.kickoff()
            st.subheader("Investment Decision")
            st.write(result)

if __name__ == '__main__':
    main()
