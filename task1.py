import re
import pandas as pd
import yfinance as yf

class MarketPredictionInputHandler:
    def __init__(self):
        self.stock_data = {}  # To store stock data for prediction

    def process_input(self, input_query):
        """
        Process the trader's input, e.g. "What happens if stock XYZ increases by 5%?"
        and extract relevant parameters: stock_symbol, percent_change, and time_frame.
        """
        pattern = r"([A-Za-z]+) increases by (\d+)%"
        match = re.search(pattern, input_query)
        
        if match:
            stock_symbol = match.group(1)
            percent_change = int(match.group(2))
            return self.get_stock_data(stock_symbol, percent_change)
        else:
            raise ValueError("Invalid input format. Please provide a query like 'stock XYZ increases by 5%'.")

    def get_stock_data(self, stock_symbol, percent_change):
        """
        Fetch historical stock data using Yahoo Finance.
        Returns data for analysis and updates the stock_data dictionary.
        """
        stock_data = yf.download(stock_symbol, period="1y", interval="1d")
        self.stock_data[stock_symbol] = stock_data
        print(f"Retrieved data for {stock_symbol}:")
        print(stock_data.tail())  # Print last few rows of stock data for preview

        return self.analyze_impact(stock_symbol, percent_change)

    def analyze_impact(self, stock_symbol, percent_change):
        """
        Analyzes the impact of the predicted price change on the portfolio.
        This method will calculate the predicted new price and estimate portfolio change.
        """
        stock_data = self.stock_data[stock_symbol]
        last_close_price = stock_data['Close'].iloc[-1]  # Get last close price
        predicted_price = last_close_price * (1 + percent_change / 100)
        
        impact = predicted_price - last_close_price
        print(f"Predicted price for {stock_symbol} after {percent_change}% change: {predicted_price}")
        print(f"Price change impact on stock: {impact}")
        
        # Return the impact value and predicted price for further calculations in prediction engine
        return {
            "stock_symbol": stock_symbol,
            "predicted_price": predicted_price,
            "impact": impact
        }

# Example of how the MarketPredictionInputHandler might be used
if __name__ == "__main__":
    handler = MarketPredictionInputHandler()
    user_input = "What happens if stock AAPL increases by 5%?"
    prediction_data = handler.process_input(user_input)
