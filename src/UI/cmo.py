#!/usr/bin/env python3
import sys
import os
import pandas as pd
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv

# Load environment variables if needed (e.g., API keys)
load_dotenv()

# Adjust the system path so that our modules can be imported if needed.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def fetch_data_yfinance(ticker_symbol, data_source='Historical', period='1y', interval='1d'):
    """
    Fetch stock data using yfinance for either Historical or Real-Time (intraday) data.

    :param ticker_symbol: The stock ticker symbol (e.g., 'AAPL').
    :param data_source: 'Historical' or 'Real-Time' (Real-Time is approximate, typically intraday quotes).
    :param period: How far back to fetch data for Historical (e.g., '1y', '6mo', etc.).
                  For Real-Time, we typically use '1d'.
    :param interval: The data interval (e.g., '1d', '1m'). For real-time (intraday), use '1m' or '5m'.
    :return: A pandas DataFrame with 'date', 'high', 'low', 'close'.
    """
    if data_source == 'Historical':
        st.info(f"Fetching historical data for {ticker_symbol} (period={period}, interval={interval})...")
        data = yf.download(ticker_symbol, period=period, interval=interval)
    else:
        st.info(f"Fetching real-time (intraday) data for {ticker_symbol} (interval={interval})...")
        # Real-time intraday data is still subject to delays via yfinance, but we'll treat it as "intraday" data
        data = yf.download(ticker_symbol, period='1d', interval=interval)
    
    if data is None or data.empty:
        st.error("Failed to fetch data using yfinance or no data returned.")
        return None

    # Reset index to make the date/time a column
    data.reset_index(inplace=True)

    # Rename columns to match the code’s expectations
    rename_map = {
        'Date': 'date',
        'Datetime': 'date',
        'Close': 'close',
        'High': 'high',
        'Low': 'low'
    }
    data.rename(columns=rename_map, inplace=True)

    # Ensure the required columns exist
    for col in ['date', 'high', 'low', 'close']:
        if col not in data.columns:
            st.error(f"Required column '{col}' not found in data.")
            return None

    return data

class CMOCalculator:
    """
    Calculates the Chande Momentum Oscillator (CMO) based on price changes over a specified period.
    
    Default Calculation:
    - Gains (gain) = price_change if > 0 else 0
    - Losses (loss) = -price_change if < 0 else 0
    - Rolling sum over 'period' for gains and losses
    - CMO = 100 * (sum_of_gains - sum_of_losses) / (sum_of_gains + sum_of_losses)
    """
    def __init__(
        self,
        df,
        period=14,
        calc_method='Standard',
        apply_smoothing=None,
        smoothing_period=3,
        keep_intermediate=False
    ):
        """
        :param df: The input DataFrame (must include 'close').
        :param period: The lookback window for the CMO calculation.
        :param calc_method: Method for calculating gains/losses. Options: 'Standard', 'Absolute'.
        :param apply_smoothing: Smoothing method for final CMO. Options: None, 'SMA', 'EMA'.
        :param smoothing_period: Period used for smoothing the CMO values.
        :param keep_intermediate: If True, retains intermediate columns after calculation.
        """
        self.df = df.copy()
        self.period = period
        self.calc_method = calc_method
        self.apply_smoothing = apply_smoothing
        self.smoothing_period = smoothing_period
        self.keep_intermediate = keep_intermediate

    def calculate(self):
        # Calculate price changes
        self.df['price_change'] = self.df['close'].diff()

        # Calculation Method
        if self.calc_method == 'Absolute':
            # Gains are absolute changes; losses are zero (or could be adapted)
            self.df['gain'] = self.df['price_change'].abs()
            self.df['loss'] = 0
        else:
            # Standard approach: split positive/negative
            self.df['gain'] = self.df['price_change'].where(self.df['price_change'] > 0, 0)
            self.df['loss'] = -self.df['price_change'].where(self.df['price_change'] < 0, 0)

        # Calculate rolling sums of gains/losses over the chosen period
        self.df['gain_sum'] = self.df['gain'].rolling(window=self.period).sum()
        self.df['loss_sum'] = self.df['loss'].rolling(window=self.period).sum()

        # Calculate the CMO
        self.df['cmo'] = 100 * (self.df['gain_sum'] - self.df['loss_sum']) / (self.df['gain_sum'] + self.df['loss_sum'])

        # Apply optional smoothing on the CMO
        if self.apply_smoothing == 'SMA':
            self.df['cmo'] = self.df['cmo'].rolling(window=self.smoothing_period).mean()
        elif self.apply_smoothing == 'EMA':
            self.df['cmo'] = self.df['cmo'].ewm(span=self.smoothing_period, adjust=False).mean()

        # Drop intermediate columns if not needed
        if not self.keep_intermediate:
            self.df.drop(columns=['price_change', 'gain', 'loss', 'gain_sum', 'loss_sum'], inplace=True)

        return self.df

def highlight_cmo_above_threshold(val, threshold):
    """Highlight cells if CMO goes above threshold (simple example styling)."""
    color = 'red' if val >= threshold else 'black'
    return f'color: {color}'

def main():
    st.title("Chande Momentum Oscillator (CMO) Calculation System")
    st.write("Fetch Real-Time or Historical Data, then calculate the CMO with various options.")

    # --- Data Source Selection ---
    data_source = st.radio(
        "Select Data Source:",
        options=["Historical", "Real-Time"],
        index=0,
        help="Use Historical data or Real-Time (intraday) data from yfinance."
    )

    if data_source == "Historical":
        # For Historical data
        period_option = st.selectbox("Historical Data Period:", options=["1y", "6mo", "3mo", "1mo"], index=0)
        interval_option = st.selectbox("Historical Interval:", options=["1d", "1wk", "1mo"], index=0)
        st.info("Historical data is retrieved for the given period and interval.")
    else:
        # For Real-Time (intraday) data
        period_option = '1d'  # Typically fixed
        interval_option = st.selectbox("Intraday Interval:", options=["1m", "5m", "15m", "30m", "1h"], index=0)
        st.warning("Real-time quotes in yfinance may be delayed, but we'll treat it as intraday data.")

    # Stock Symbol
    ticker_symbol = st.text_input("Enter Stock Symbol:", value="AAPL")

    # --- CMO Customization ---
    st.subheader("CMO Parameters")
    cmo_period = st.number_input("CMO Calculation Period:", min_value=1, max_value=200, value=14)

    calc_method = st.selectbox(
        "Gains/Loss Calculation Method:",
        options=["Standard", "Absolute"],
        index=0
    )

    apply_smoothing = st.selectbox(
        "Apply Smoothing to CMO?",
        options=[None, "SMA", "EMA"],
        format_func=lambda x: "None" if x is None else x,
        index=0
    )
    smoothing_period = st.number_input("Smoothing Period (for SMA/EMA):", min_value=1, max_value=50, value=3)

    keep_intermediate = st.checkbox("Keep intermediate columns for debugging?", value=False)

    threshold_enable = st.checkbox("Enable threshold highlight on CMO?", value=False)
    threshold_value = st.number_input("Highlight CMO above threshold value:", value=70) if threshold_enable else None

    # --- Buttons ---
    if st.button("Fetch & Calculate CMO"):
        data = fetch_data_yfinance(
            ticker_symbol=ticker_symbol,
            data_source=data_source,
            period=period_option,
            interval=interval_option
        )

        if data is not None:
            st.subheader(f"Fetched Data for {ticker_symbol} ({data_source})")
            st.dataframe(data.tail(10))

            # Calculate the CMO
            cmo_calc = CMOCalculator(
                df=data,
                period=cmo_period,
                calc_method=calc_method,
                apply_smoothing=apply_smoothing,
                smoothing_period=smoothing_period,
                keep_intermediate=keep_intermediate
            )
            cmo_data = cmo_calc.calculate()

            st.subheader("Calculated CMO Data")
            if threshold_enable and threshold_value is not None:
                styled_cmo_data = cmo_data.style.applymap(
                    lambda val: highlight_cmo_above_threshold(val, threshold_value),
                    subset=['cmo']  # Apply styling only to the 'cmo' column
                )
                st.dataframe(styled_cmo_data)
            else:
                st.dataframe(cmo_data.tail(20))

if __name__ == '__main__':
    main()
