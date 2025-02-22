#!/usr/bin/env python3
import sys
import os
import pandas as pd
import streamlit as st
from yahooquery import Ticker
from dotenv import load_dotenv

# Load environment variables if needed (e.g., API keys)
load_dotenv()

# Adjust the system path so that our modules can be imported.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
    else:
        st.error("Failed to fetch data as a DataFrame.")
        return None

    # Ensure required columns exist; rename if necessary.
    for col in ['date', 'high', 'low', 'close']:
        if col not in data.columns:
            # Try converting from capitalized names if needed.
            if col.capitalize() in data.columns:
                data.rename(columns={col.capitalize(): col}, inplace=True)
            else:
                st.error(f"Required column '{col}' not found in data.")
                return None
    return data

class CMOCalculator:
    """
    Calculates the Chande Momentum Oscillator (CMO) based on price changes over specified periods.
    Formula: CMO = (Sum of Gains for n periods) - (Sum of Losses for n periods) / (Sum of Gains for n periods) + (Sum of Losses for n periods)
    """
    def __init__(self, df, period=14):
        self.df = df.copy()
        self.period = period

    def calculate(self):
        # Calculate price changes
        self.df['price_change'] = self.df['close'].diff()

        # Calculate gains and losses
        self.df['gain'] = self.df['price_change'].where(self.df['price_change'] > 0, 0)
        self.df['loss'] = -self.df['price_change'].where(self.df['price_change'] < 0, 0)

        # Calculate the rolling sums of gains and losses over the specified period
        self.df['gain_sum'] = self.df['gain'].rolling(window=self.period).sum()
        self.df['loss_sum'] = self.df['loss'].rolling(window=self.period).sum()

        # Calculate the CMO
        self.df['cmo'] = 100 * (self.df['gain_sum'] - self.df['loss_sum']) / (self.df['gain_sum'] + self.df['loss_sum'])

        # Drop the intermediate columns for clean display
        self.df.drop(columns=['price_change', 'gain', 'loss', 'gain_sum', 'loss_sum'], inplace=True)

        return self.df

def main():
    st.title("Chande Momentum Oscillator (CMO) Calculation System")
    st.write("Calculate the Chande Momentum Oscillator (CMO) for your selected stock.")

    # Input for stock symbol and data period.
    ticker_symbol = st.text_input("Enter Stock Symbol:", value="AAPL")
    period_option = st.selectbox("Select Data Period:", options=["1y", "6mo", "3mo", "1mo"], index=0)

    # Input for CMO calculation period.
    period = st.number_input("CMO Calculation Period:", min_value=1, max_value=200, value=14)

    # Button to calculate and display CMO data.
    if st.button("Calculate CMO"):
        # Fetch the historical data.
        data = fetch_stock_data(ticker_symbol, period=period_option)
        if data is not None:
            st.subheader(f"Original Stock Data for {ticker_symbol}")
            st.dataframe(data.tail(10))

            # Calculate the CMO
            cmo_calc = CMOCalculator(data, period=period)
            cmo_data = cmo_calc.calculate()

            st.subheader("Calculated CMO Data")
            st.dataframe(cmo_data.tail(20))

    # Button to fetch and display the latest data.
    if st.button("Fetch Latest Data"):
        latest_data = fetch_stock_data(ticker_symbol, period=period_option)
        if latest_data is not None:
            st.subheader(f"Latest Stock Data for {ticker_symbol}")
            st.dataframe(latest_data.tail(10))

if __name__ == '__main__':
    main()
