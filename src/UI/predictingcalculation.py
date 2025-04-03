import json
from datetime import datetime, timedelta

import streamlit as st
import yfinance as yf
from crewai import Agent, Crew, Task

from src.Agents.input import process_user_input  # Import Week 1 function


# Function to interpret the time frame for Yahoo Finance
def get_time_frame(query_time_frame):
    try:
        if "end of this month" in query_time_frame.lower():
            today = datetime.today()
            end_of_month = today.replace(day=28) + timedelta(days=4)  # Get the last day of the month
            return today, end_of_month
        elif "next year" in query_time_frame.lower():
            today = datetime.today()
            next_year = today.replace(year=today.year + 1)
            return today, next_year
        elif "6 months" in query_time_frame.lower():
            today = datetime.today()
            six_months_later = today + timedelta(days=180)  # Roughly 6 months
            return today, six_months_later
        else:
            today = datetime.today()
            thirty_days_later = today + timedelta(days=30)
            return today, thirty_days_later
    except Exception as e:
        print(f"Error processing time frame: {e}")
        return None, None

# Function to fetch stock data and calculate the predicted value based on percentage change
def get_stock_prediction(stock_symbol, percentage_change, start_date, end_date):
    try:
        stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
        last_price = stock_data['Close'].iloc[-1]  # Last available closing price
        
        stock_info = yf.Ticker(stock_symbol).info
        total_shares_outstanding = stock_info.get('sharesOutstanding', 0)
        
        predicted_price = last_price * (1 + (percentage_change / 100))
        
        initial_market_value = last_price * total_shares_outstanding
        final_market_value = predicted_price * total_shares_outstanding
        
        return {
            "stock_symbol": stock_symbol,
            "last_price": last_price,
            "predicted_price": predicted_price,
            "initial_market_value": initial_market_value,
            "final_market_value": final_market_value,
        }
    except Exception as e:
        print(f"Error fetching stock data for {stock_symbol}: {e}")
        return None

# CrewAI Agents
input_agent = Agent(
    role="Input Extraction Agent",
    goal="Extract relevant stock market details from user input",
    verbose=True,
    allow_delegation=False
)

time_frame_agent = Agent(
    role="Time Frame Processing Agent",
    goal="Convert user-defined time frames to a Yahoo Finance-compatible format",
    verbose=True
)

stock_data_agent = Agent(
    role="Stock Data Fetcher",
    goal="Fetch stock market data and calculate market values",
    verbose=True
)

output_agent = Agent(
    role="Output Display Agent",
    goal="Present the final prediction in a structured format",
    verbose=True
)

# Define CrewAI Tasks
def run_stock_prediction(user_query):
    extracted_data = process_user_input(user_query)

    if not extracted_data:
        return {"error": "Failed to extract stock details."}

    stock_symbol = extracted_data.get("stock_symbol")
    percentage_change = extracted_data.get("percentage_change")
    time_frame = extracted_data.get("time_frame")

    if not (stock_symbol and percentage_change is not None and time_frame):
        return {"error": "Incomplete data for stock prediction."}

    start_date, end_date = get_time_frame(time_frame)

    if not (start_date and end_date):
        return {"error": "Invalid time frame format."}

    return get_stock_prediction(stock_symbol, percentage_change, start_date, end_date)

# Streamlit UI
def main():
    st.title("📈 Stock Market Prediction")
    st.write("Enter your stock market prediction query and get future projections.")

    user_query = st.text_input("Enter your market prediction query (e.g., 'What will be the market value of AAPL if it decreases by 5% by the end of this month?')")

    if st.button("Predict"):
        with st.spinner("Processing..."):
            result = run_stock_prediction(user_query)

        if "error" in result:
            st.error(result["error"])
        else:
            st.success("Prediction Successful!")
            st.write("### 📊 Prediction Results")
            st.json(result)

            # Display key information
            st.write(f"**Stock Symbol:** {result['stock_symbol']}")
            st.write(f"**Last Closing Price:** ${result['last_price']:.2f}")
            st.write(f"**Predicted Price:** ${result['predicted_price']:.2f}")
            st.write(f"**Initial Market Value:** ${result['initial_market_value']:.2f}")
            st.write(f"**Final Market Value:** ${result['final_market_value']:.2f}")

            # Plotting stock prices
            chart_data = {
                "Time": ["Current", "Predicted"],
                "Stock Price": [result["last_price"], result["predicted_price"]],
            }

            st.line_chart(chart_data, x="Time", y="Stock Price")

if __name__ == "__main__":
    main()
