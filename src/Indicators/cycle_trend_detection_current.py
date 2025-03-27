import numpy as np
import pandas as pd
import math
import yfinance as yf
import os
from datetime import datetime, timedelta

class CycleDetector:
    def __init__(self, symbol='SPY', start_date=None, end_date=None, file_path=None,
                 lower_bound=18, upper_bound=40, length=40, window=10, stability_threshold=5):
        # Get the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # If no file_path is provided, construct it using the symbol
        if file_path is None:
            file_path = os.path.join(script_dir, f"{symbol}_data.xlsx")
        
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.file_path = file_path
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.length = length
        self.window = window
        self.stability_threshold = stability_threshold
        self.mu = 1 / self.length
        self.df = None

    def fetch_data(self):
        # Print debug information
        print(f"Attempting to load data from: {self.file_path}")
        
        if self.file_path and os.path.exists(self.file_path):
            return self.load_from_excel()
        elif self.symbol:
            return self.load_from_yfinance()
        else:
            raise ValueError("No valid data source provided.")

    def load_from_excel(self):
        try:
            # Read the Excel file
            self.df = pd.read_excel(self.file_path)
            
            # Ensure Date column is in datetime format
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            
            # Validate data
            if len(self.df) < self.length:
                raise ValueError(f"Not enough data points. Need at least {self.length} rows.")
            
            # Try to find the close price column (case-insensitive)
            close_columns = [col for col in self.df.columns if col.lower() == 'close']
            
            if not close_columns:
                # If no 'Close' column found, print all columns and raise an error
                print("Available columns:", list(self.df.columns))
                raise ValueError("No 'Close' column found in the Excel file.")
            
            # Convert to list and return
            return self.df[close_columns[0]].tolist()
        except Exception as e:
            print(f"Error reading Excel file: {e}")
            raise

    def highpass_filter(self, price_series, period):
        # Ensure enough data points
        if len(price_series) < 3:
            return price_series

        a1 = math.exp(-1.414 * math.pi / period)
        b1 = 2 * a1 * math.cos(1.414 * 180 / period)
        c1, c2, c3 = (1 + b1) / 4, b1, -(a1 ** 2)
        hp_series = [0.0, 0.0]
        for i in range(2, len(price_series)):
            hp_series.append(c1 * (price_series[i] - 2 * price_series[i - 1] + price_series[i - 2]) + c2 * hp_series[i - 1] + c3 * hp_series[i - 2])
        return hp_series
    
    def super_smoother(self, price_series, period):
        # Ensure enough data points
        if len(price_series) < 3:
            return price_series

        a1 = math.exp(-1.414 * math.pi / period)
        b1 = 2 * a1 * math.cos(1.414 * 180 / period)
        c1, c2, c3 = 1 - b1 + a1**2, b1, -(a1**2)
        ss_series = [price_series[0], price_series[1]]
        for i in range(2, len(price_series)):
            ss_series.append(c1 * (price_series[i] + price_series[i - 1]) / 2 + c2 * ss_series[i - 1] + c3 * ss_series[i - 2])
        return ss_series

    def get_current_trend(self, lookback_periods=None):
        """
        Determine the current trend status with robust error handling
        """
        # Ensure data is loaded
        if self.df is None:
            self.fetch_data()
        
        # Use default lookback if not specified
        if lookback_periods is None:
            lookback_periods = min(self.length, len(self.df))
        
        # Get the most recent close prices
        recent_prices = self.df['Close'].tail(lookback_periods).tolist()
        
        # Basic trend detection using price movement
        price_changes = np.diff(recent_prices)
        
        # Simple trend detection if cycle detection fails
        try:
            # Perform cycle detection on recent prices
            hp = self.highpass_filter(recent_prices, self.upper_bound)
            lp = self.super_smoother(hp, self.lower_bound)
            
            # Detect cycles in the recent period
            recent_cycles = []
            for i in range(max(0, len(lp) - self.length)):
                segment = lp[i : i + min(self.length, len(lp) - i)]
                max_power, dominant_cycle = 0, 0
                for period in range(self.lower_bound, self.upper_bound + 1):
                    real_part = sum(segment[j] * math.cos(2 * math.pi * j / period) for j in range(len(segment)))
                    imag_part = sum(segment[j] * math.sin(2 * math.pi * j / period) for j in range(len(segment)))
                    power = 0.1 / ((1 - real_part) ** 2 + imag_part ** 2) if (1 - real_part) ** 2 + imag_part ** 2 != 0 else 0
                    if power > max_power:
                        max_power, dominant_cycle = power, period
                recent_cycles.append(dominant_cycle)
            
            # Classify trend
            if recent_cycles:
                window_cycles = recent_cycles[-self.window:]
                cycle_variability = max(window_cycles) - min(window_cycles)
                is_trending = cycle_variability > self.stability_threshold
            else:
                # Fallback to price movement trend detection
                is_trending = np.std(price_changes) > np.mean(np.abs(price_changes))
        
        except Exception as e:
            # Fallback to simple trend detection
            print(f"Cycle detection failed: {e}")
            is_trending = np.std(price_changes) > np.mean(np.abs(price_changes))
        
        # Prepare trend analysis
        trend_analysis = {
            'is_trending': is_trending,
            'trend_type': 'Trending' if is_trending else 'Non-Trending',
            'current_date': self.df['Date'].iloc[-1],
            'current_price': self.df['Close'].iloc[-1],
            'price_changes': price_changes.tolist() if len(price_changes) > 0 else []
        }
        
        return trend_analysis

# Example usage
if __name__ == "__main__":
    try:
        # Initialize the detector
        detector = CycleDetector()
        
        # Get current trend
        current_trend = detector.get_current_trend()
        
        # Print detailed trend information
        print("\nCurrent Trend Analysis:")
        print(f"Trend Status: {current_trend['trend_type']}")
        print(f"Current Date: {current_trend['current_date'].date()}")
        print(f"Current Price: ${current_trend['current_price']:.2f}")
        print(f"Is Trending: {current_trend['is_trending']}")
        
    except Exception as e:
        print(f"Error: {e}")