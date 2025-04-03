#!/usr/bin/env python3
"""
This Streamlit application fetches historical stock data and computes various risk metrics.
It also performs portfolio risk breakdown analysis and simulates scenario analysis.
CrewAI integration sections have been removed so that the application only fetches and analyzes data.
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from yahooquery import Ticker
from dotenv import load_dotenv
from textwrap import dedent
from datetime import datetime  # For dynamically injecting current date into outputs

# ------------------------------
# Setup and Imports
# ------------------------------
# Update system path to import custom modules from parent directories if necessary.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import custom tools (even if not used, they are available for future extensions).
from src.Agents.Analysis.Tools.browser_tools import BrowserTools
from src.Agents.Analysis.Tools.calculator_tools import CalculatorTools
from src.Agents.Analysis.Tools.search_tools import SearchTools
from src.Agents.Analysis.Tools.sec_tools import SECTools
from langchain_community.tools import YahooFinanceNewsTool

# Load environment variables (e.g., API keys) from a .env file.
load_dotenv()

# ------------------------------
# Data Fetching Functions
# ------------------------------
def fetch_stock_data(ticker_symbol, period='1y'):
    """
    Fetch historical stock data for a given ticker symbol using yahooquery.

    Parameters:
      ticker_symbol (str): The stock ticker symbol (e.g., "AAPL").
      period (str): The period over which to fetch historical data (e.g., "1y" for one year).

    Returns:
      DataFrame: A pandas DataFrame containing historical data with columns such as 'date', 'high', 'low', and 'close'.
    
    Process:
      1. Create a Ticker instance using yahooquery.
      2. Call the history method with the specified period.
      3. Reset the index so that 'date' becomes a column.
      4. Convert the 'date' column to datetime (removing timezone info).
      5. Verify that all required columns are present; rename columns if needed.
    """
    st.info(f"Fetching historical data for {ticker_symbol} (Period: {period})...")
    ticker = Ticker(ticker_symbol)  # Create Ticker instance for the given symbol.
    data = ticker.history(period=period)  # Fetch historical data.
    
    # Check if data is returned as a DataFrame.
    if isinstance(data, pd.DataFrame):
        data.reset_index(inplace=True)  # Reset index to ensure 'date' is a column.
        # Convert the 'date' column to datetime and remove any timezone information.
        data['date'] = pd.to_datetime(data['date'], utc=True).dt.tz_convert(None)
    else:
        st.error("Failed to fetch data as a DataFrame.")
        return None

    # Verify that required columns exist; if they are capitalized, rename them to lowercase.
    for col in ['date', 'high', 'low', 'close']:
        if col not in data.columns:
            if col.capitalize() in data.columns:
                data.rename(columns={col.capitalize(): col}, inplace=True)
            else:
                st.error(f"Required column '{col}' not found in data.")
                return None
    return data

def fetch_current_stock_price(ticker_symbol):
    """
    Fetch the current stock price for a given ticker symbol using yahooquery.

    This function attempts to retrieve the current price from the 'price' attribute.
    If that fails, it falls back to the 'summary_detail' attribute.
    The uppercase version of the ticker symbol is used as a key when necessary.

    Parameters:
      ticker_symbol (str): The stock ticker (e.g., "AAPL").

    Returns:
      float: The current stock price, or None if it cannot be retrieved.
    
    Process:
      1. Create a Ticker instance.
      2. Try to get the current price from ticker.price.
      3. If not available, attempt to get it from ticker.summary_detail.
      4. Report an error if both methods fail.
    """
    st.info(f"Fetching current stock price for {ticker_symbol}...")
    ticker = Ticker(ticker_symbol)
    try:
        realtime_data = ticker.price  # Attempt to get real-time price data.
        key = ticker_symbol.upper()  # Use uppercase ticker as key.
        if isinstance(realtime_data, dict):
            if key in realtime_data and isinstance(realtime_data[key], dict) and "regularMarketPrice" in realtime_data[key]:
                return realtime_data[key]["regularMarketPrice"]
            elif "regularMarketPrice" in realtime_data:
                return realtime_data["regularMarketPrice"]
        # If not found in 'price', try the 'summary_detail' attribute.
        summary = ticker.summary_detail
        if isinstance(summary, dict):
            if key in summary and isinstance(summary[key], dict) and "regularMarketPrice" in summary[key]:
                return summary[key]["regularMarketPrice"]
            elif "regularMarketPrice" in summary:
                return summary["regularMarketPrice"]
        st.error("Failed to fetch current stock price from yahooquery.")
        return None
    except Exception as e:
        st.error(f"Error fetching current stock price: {e}")
        return None

# ------------------------------
# Risk Metrics Calculation Functions
# ------------------------------
def calculate_risk_metrics(data, confidence=0.05):
    """
    Calculate key risk metrics from historical stock data.

    Metrics calculated include:
      - Daily returns: Percentage change in closing prices.
      - Value at Risk (VaR): The potential loss as a percentage at a given confidence level.
      - Maximum drawdown: The largest decline in cumulative returns from a peak.
      - Annualized volatility: The standard deviation of daily returns scaled to an annual figure.

    Parameters:
      data (DataFrame): The historical stock data.
      confidence (float): The confidence level for VaR (e.g., 0.05 for 5%).

    Returns:
      tuple: A tuple containing:
             (1) A dictionary of calculated risk metrics, and
             (2) The updated DataFrame with additional columns (returns, cumulative_return, drawdown).

    Process:
      1. Sort the data by date.
      2. Compute daily returns using percentage change.
      3. Drop missing return values.
      4. Calculate VaR using the percentile method.
      5. Compute cumulative returns.
      6. Calculate the running maximum and then the drawdown.
      7. Annualize the volatility.
    """
    data = data.sort_values(by='date').copy()
    data['returns'] = data['close'].pct_change()
    returns = data['returns'].dropna()
    
    # Compute VaR as the specified percentile of the returns.
    var_value = np.percentile(returns, confidence * 100)
    
    # Compute cumulative returns.
    data['cumulative_return'] = (1 + returns).cumprod()
    # Compute the running maximum of cumulative returns.
    data['running_max'] = data['cumulative_return'].cummax()
    # Calculate drawdown as the percentage decline from the running maximum.
    data['drawdown'] = (data['cumulative_return'] - data['running_max']) / data['running_max']
    max_drawdown = data['drawdown'].min()
    
    # Calculate annualized volatility (assuming 252 trading days per year).
    volatility = returns.std() * np.sqrt(252)
    
    risk_metrics = {
        "var": var_value,
        "max_drawdown": max_drawdown,
        "volatility": volatility
    }
    return risk_metrics, data

def calculate_scenario_risk_metrics(data, shock, confidence=0.05):
    """
    Calculate risk metrics under a simulated market shock.
    A fixed shock (e.g., a 5% drop) is subtracted from the daily returns, and new risk metrics are computed.

    Parameters:
      data (DataFrame): The historical stock data.
      shock (float): The shock value to subtract from daily returns (e.g., 0.05 for a 5% drop).
      confidence (float): The confidence level for VaR calculation.

    Returns:
      dict: A dictionary containing risk metrics computed on the "shocked" returns.
    
    Process:
      1. Compute daily returns.
      2. Subtract the shock value from the returns.
      3. Recalculate cumulative returns, drawdown, and volatility using the shocked returns.
    """
    data = data.sort_values(by='date').copy()
    data['returns'] = data['close'].pct_change()
    data['shock_returns'] = data['returns'] - shock
    shock_returns = data['shock_returns'].dropna()
    
    var_value = np.percentile(shock_returns, confidence * 100)
    data['cumulative_return'] = (1 + shock_returns).cumprod()
    data['running_max'] = data['cumulative_return'].cummax()
    data['drawdown'] = (data['cumulative_return'] - data['running_max']) / data['running_max']
    max_drawdown = data['drawdown'].min()
    volatility = shock_returns.std() * np.sqrt(252)
    
    scenario_metrics = {
        "var": var_value,
        "max_drawdown": max_drawdown,
        "volatility": volatility
    }
    return scenario_metrics

# ------------------------------
# Portfolio Breakdown Analysis
# ------------------------------
def analyze_portfolio_breakdown(portfolio_str, period='1y', confidence=0.05):
    """
    Parse a multi-line portfolio input string and compute risk metrics for each position.

    Expected input format (one per line):
      "ticker, asset_class, position_size"

    Returns:
      tuple: A tuple containing:
             (1) A detailed DataFrame with risk metrics for each individual position.
             (2) A grouped DataFrame with weighted risk metrics aggregated by asset class.

    Process:
      1. Split the input string into lines.
      2. For each line, split into ticker, asset class, and position size.
      3. Fetch historical data and compute risk metrics.
      4. Collect results in a list and then create a DataFrame.
      5. Group the DataFrame by asset class and compute weighted averages.
    """
    lines = portfolio_str.strip().splitlines()
    records = []
    for line in lines:
        try:
            # Split the line into its components.
            ticker, asset_class, position_size = [x.strip() for x in line.split(",")]
            position_size = float(position_size)
            # Fetch historical stock data for the ticker.
            data = fetch_stock_data(ticker, period)
            if data is None:
                continue
            # Calculate risk metrics for this ticker.
            metrics, _ = calculate_risk_metrics(data, confidence)
            records.append({
                "Ticker": ticker,
                "Asset Class": asset_class,
                "Position Size": position_size,
                "VaR": metrics["var"],
                "Max Drawdown": metrics["max_drawdown"],
                "Volatility": metrics["volatility"]
            })
        except Exception as e:
            st.error(f"Error processing line '{line}': {e}")
    
    if records:
        # Create a DataFrame from the collected records.
        df = pd.DataFrame(records)
        # Group by asset class and compute weighted averages.
        grouped = df.groupby("Asset Class").apply(
            lambda g: pd.Series({
                "Total Position": g["Position Size"].sum(),
                "Weighted VaR": np.average(g["VaR"], weights=g["Position Size"]),
                "Weighted Max Drawdown": np.average(g["Max Drawdown"], weights=g["Position Size"]),
                "Weighted Volatility": np.average(g["Volatility"], weights=g["Position Size"])
            })
        ).reset_index()
        return df, grouped
    else:
        return None, None

# ------------------------------
# Visualization Functions
# ------------------------------
def plot_price_chart(data, ticker_symbol):
    """
    Plot the historical closing price of a stock.

    Parameters:
      data (DataFrame): Historical stock data.
      ticker_symbol (str): The stock ticker.
    """
    fig, ax = plt.subplots()
    ax.plot(data['date'], data['close'], label='Close Price')
    ax.set_title(f"{ticker_symbol} Price Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

def plot_drawdown_chart(data, ticker_symbol):
    """
    Plot the drawdown (decline from peak) of cumulative returns.

    Parameters:
      data (DataFrame): Stock data with calculated drawdown.
      ticker_symbol (str): The stock ticker.
    """
    fig, ax = plt.subplots()
    ax.plot(data['date'], data['drawdown'], label='Drawdown', color='red')
    ax.set_title(f"{ticker_symbol} Drawdown Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.legend()
    st.pyplot(fig)

def plot_return_histogram(returns, var_value):
    """
    Plot a histogram of daily returns with a vertical line indicating the VaR level.

    Parameters:
      returns (Series): A pandas Series of daily returns.
      var_value (float): The Value at Risk (VaR) level to highlight.
    """
    fig, ax = plt.subplots()
    ax.hist(returns, bins=50, alpha=0.7, label='Daily Returns')
    ax.axvline(var_value, color='red', linestyle='dashed', linewidth=2,
               label=f'VaR ({var_value:.2%})')
    ax.set_title("Histogram of Daily Returns")
    ax.set_xlabel("Return")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

# ------------------------------
# Main Streamlit Application
# ------------------------------
def main():
    # Set the title and introductory text for the dashboard.
    st.title("Portfolio Risk Assessment Dashboard")
    st.write(
        "Visualize portfolio risk metrics, get detailed breakdowns by asset class and position, "
        "and validate risk metrics under simulated market scenarios."
    )
    
    # ------------------------------
    # Sidebar: Single Ticker Analysis Inputs
    # ------------------------------
    st.sidebar.header("Single Ticker Analysis")
    # Allow the user to input a stock ticker (default is AAPL).
    ticker_symbol = st.sidebar.text_input("Enter Stock Symbol:", value="AAPL")
    # Provide a dropdown to select the data period.
    period_option = st.sidebar.selectbox("Select Data Period:", options=["1y", "6mo", "3mo", "1mo"], index=0)
    # Provide a slider to set the VaR confidence level.
    confidence_level = st.sidebar.slider(
        "VaR Confidence Level",
        min_value=0.01, max_value=0.1, value=0.05, step=0.01,
        help="Lower tail probability for VaR calculation (e.g., 0.05 for 5%)"
    )
    
    # ------------------------------
    # Section 1: Single Ticker Analysis
    # ------------------------------
    if st.button("Fetch and Analyze Data"):
        # Fetch historical data for the specified ticker and period.
        data = fetch_stock_data(ticker_symbol, period=period_option)
        if data is not None:
            st.subheader(f"Historical Data for {ticker_symbol}")
            st.dataframe(data.tail(10))  # Display the 10 most recent records.
            # Visualize the historical price chart.
            plot_price_chart(data, ticker_symbol)
            
            # Calculate risk metrics based on the historical data.
            risk_metrics, risk_data = calculate_risk_metrics(data, confidence=confidence_level)
            st.subheader("Calculated Risk Metrics")
            st.markdown(f"- **Value at Risk (VaR) at {confidence_level*100:.0f}% level:** {risk_metrics['var']:.2%}")
            st.markdown(f"- **Maximum Drawdown:** {risk_metrics['max_drawdown']:.2%}")
            st.markdown(f"- **Annualized Volatility:** {risk_metrics['volatility']:.2%}")
            
            # Visualize the drawdown chart.
            plot_drawdown_chart(risk_data, ticker_symbol)
            # Plot the histogram of daily returns.
            returns = risk_data['returns'].dropna()
            plot_return_histogram(returns, risk_metrics['var'])
            
            # Store computed risk metrics and raw data in session state for later use.
            st.session_state.risk_metrics = risk_metrics
            st.session_state.single_data = data

    # ------------------------------
    # Section 2: Portfolio Risk Breakdown Analysis
    # ------------------------------
    st.header("Portfolio Risk Breakdown")
    st.write(
        "Enter your portfolio positions in the format: **Ticker, Asset Class, Position Size** (one per line). For example:\n\n"
        "`AAPL, Equity, 100`\n`MSFT, Equity, 150`\n`TLT, Bond, 200`\n`GLD, Commodity, 50`"
    )
    # Provide a text area for users to input multiple portfolio positions.
    portfolio_input = st.text_area("Portfolio Positions", 
        value="AAPL, Equity, 100\nMSFT, Equity, 150\nTLT, Bond, 200\nGLD, Commodity, 50", height=150)
    if st.button("Analyze Portfolio Breakdown"):
        # Analyze the provided portfolio positions and compute risk metrics.
        details_df, breakdown_df = analyze_portfolio_breakdown(portfolio_input, period=period_option, confidence=confidence_level)
        if details_df is not None:
            st.subheader("Individual Position Risk Metrics")
            st.dataframe(details_df)
            st.subheader("Risk Breakdown by Asset Class (Weighted Averages)")
            st.dataframe(breakdown_df)
            # Save the grouped breakdown in session state for any further analysis.
            st.session_state.breakdown_df = breakdown_df
        else:
            st.error("No valid portfolio positions found or error in processing.")
    
    # ------------------------------
    # Section 3: Scenario Analysis
    # ------------------------------
    st.header("Scenario Analysis: Validate Risk Metrics")
    # Slider to select a simulated market shock percentage.
    shock_percent = st.slider(
        "Simulated Market Shock (%)",
        min_value=0.0, max_value=10.0, value=2.0, step=0.5,
        help="Enter the percentage shock to subtract from daily returns (e.g., 2% shock)."
    )
    if st.button("Validate Risk Metrics Under Scenario"):
        if "single_data" not in st.session_state:
            st.error("Please perform the single ticker analysis first.")
        else:
            # Convert shock percentage to a decimal value.
            shock = shock_percent / 100.0
            # Recalculate baseline risk metrics using the stored single ticker data.
            base_metrics, base_data = calculate_risk_metrics(st.session_state.single_data, confidence=confidence_level)
            # Calculate risk metrics under the simulated shock.
            scenario_metrics = calculate_scenario_risk_metrics(st.session_state.single_data, shock, confidence=confidence_level)
            st.subheader("Baseline vs. Scenario Risk Metrics")
            # Create a DataFrame to compare the baseline and shocked metrics side-by-side.
            comparison = pd.DataFrame({
                "Metric": ["VaR", "Max Drawdown", "Annualized Volatility"],
                "Baseline": [f"{base_metrics['var']:.2%}", f"{base_metrics['max_drawdown']:.2%}", f"{base_metrics['volatility']:.2%}"],
                "Scenario (Shock -{0:.0%})".format(shock): [f"{scenario_metrics['var']:.2%}", f"{scenario_metrics['max_drawdown']:.2%}", f"{scenario_metrics['volatility']:.2%}"]
            })
            st.dataframe(comparison)
    
if __name__ == '__main__':
    main()
