import numpy as np
import pandas as pd
import math
import yfinance as yf
import os

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
        self.df = None  # To store the full dataframe

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
            
            # Print column names for debugging
            print("Columns in the Excel file:", list(self.df.columns))
            
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

    def load_from_yfinance(self):
        try:
            df = yf.download(self.symbol, start=self.start_date, end=self.end_date)
            if df.empty:
                raise ValueError("Yahoo Finance returned no data.")
            return df['Close'].tolist()
        except Exception as e:
            raise ValueError(f"Error fetching data from Yahoo Finance: {e}")
        

    def highpass_filter(self, price_series, period):
        a1 = math.exp(-1.414 * math.pi / period)
        b1 = 2 * a1 * math.cos(1.414 * 180 / period)
        c1, c2, c3 = (1 + b1) / 4, b1, -(a1 ** 2)
        hp_series = [0.0, 0.0]
        for i in range(2, len(price_series)):
            hp_series.append(c1 * (price_series[i] - 2 * price_series[i - 1] + price_series[i - 2]) + c2 * hp_series[i - 1] + c3 * hp_series[i - 2])
        return hp_series
    
    def super_smoother(self, price_series, period):
        a1 = math.exp(-1.414 * math.pi / period)
        b1 = 2 * a1 * math.cos(1.414 * 180 / period)
        c1, c2, c3 = 1 - b1 + a1**2, b1, -(a1**2)
        ss_series = [price_series[0], price_series[1]]
        for i in range(2, len(price_series)):
            ss_series.append(c1 * (price_series[i] + price_series[i - 1]) / 2 + c2 * ss_series[i - 1] + c3 * ss_series[i - 2])
        return ss_series
    
    def detect_cycles(self, close_prices):
        hp = self.highpass_filter(close_prices, self.upper_bound)
        lp = self.super_smoother(hp, self.lower_bound)
        dominant_cycles = []
        for i in range(len(lp) - self.length):
            segment = lp[i : i + self.length]
            max_power, dominant_cycle = 0, 0
            for period in range(self.lower_bound, self.upper_bound + 1):
                real_part = sum(segment[j] * math.cos(2 * math.pi * j / period) for j in range(self.length))
                imag_part = sum(segment[j] * math.sin(2 * math.pi * j / period) for j in range(self.length))
                power = 0.1 / ((1 - real_part) ** 2 + imag_part ** 2) if (1 - real_part) ** 2 + imag_part ** 2 != 0 else 0
                if power > max_power:
                    max_power, dominant_cycle = power, period
            dominant_cycles.append(dominant_cycle)
        return dominant_cycles
    
    def classify_trend(self, dominant_cycles):
        trend_labels = []
        for i in range(len(dominant_cycles) - self.window):
            window_cycles = dominant_cycles[i : i + self.window]
            cycle_variability = max(window_cycles) - min(window_cycles)
            trend_labels.append('Trending' if cycle_variability > self.stability_threshold else 'Non-Trending')
        return trend_labels

    def run(self):
        close_prices = self.fetch_data()
        if not close_prices:
            return "No data available."
        
        dominant_cycles = self.detect_cycles(close_prices)
        trend_classification = self.classify_trend(dominant_cycles)
        
        # Prepare the results with date ranges
        return self.analyze_trend_periods(trend_classification)

    def analyze_trend_periods(self, trend_labels):
        # Adjust for the window and length differences
        start_index = self.length
        dates = self.df['Date'].tolist()[start_index:]
        
        # Combine dates with trend labels
        trend_periods = []
        current_trend = trend_labels[0]
        current_start = dates[0]
        
        for i in range(1, len(trend_labels)):
            if trend_labels[i] != current_trend:
                # Trend changed, log the previous period
                trend_periods.append({
                    'trend': current_trend,
                    'start_date': current_start,
                    'end_date': dates[i-1],
                    'duration': (dates[i-1] - current_start).days
                })
                # Start a new period
                current_trend = trend_labels[i]
                current_start = dates[i]
        
        # Add the last period
        trend_periods.append({
            'trend': current_trend,
            'start_date': current_start,
            'end_date': dates[-1],
            'duration': (dates[-1] - current_start).days
        })
        
        return trend_periods

# Example usage
if __name__ == "__main__":
    try:
        # By default, it will look for SPY_data.xlsx in the script's directory
        detector = CycleDetector()
        trend_periods = detector.run()
        
        # Print results in a readable format
        print("\nTrend Analysis:")
        for period in trend_periods:
            print(f"Trend: {period['trend']}")
            print(f"  Start Date: {period['start_date'].date()}")
            print(f"  End Date: {period['end_date'].date()}")
            print(f"  Duration: {period['duration']} days")
            print()
    except Exception as e:
        print(f"Error: {e}")
