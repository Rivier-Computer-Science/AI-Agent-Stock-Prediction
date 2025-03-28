# src/Indicators/trend_detector_combined.py
import numpy as np
import pandas as pd
import math
import yfinance as yf
import os
from datetime import datetime, timedelta
from src.Indicators.usi_jg import calculate_usi  # Assuming this is your USI function

class TrendDetector:
    def __init__(self, symbol='SPY', start_date=None, end_date=None, file_path=None,
                 usi_length=28, usi_window=4, usi_trend_threshold=0.5,
                 griffiths_lower_bound=18, griffiths_upper_bound=40, griffiths_length=40,
                 griffiths_window=10, griffiths_stability_threshold=5):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if file_path is None:
            file_path = os.path.join(script_dir, f"{symbol}_data.xlsx")

        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.file_path = file_path
        # USI parameters
        self.usi_length = usi_length
        self.usi_window = usi_window
        self.usi_trend_threshold = usi_trend_threshold
        # Griffiths parameters
        self.griffiths_lower_bound = griffiths_lower_bound
        self.griffiths_upper_bound = griffiths_upper_bound
        self.griffiths_length = griffiths_length
        self.griffiths_window = griffiths_window
        self.griffiths_stability_threshold = griffiths_stability_threshold
        self.df = None

    def fetch_data(self):
        print(f"Attempting to load data from: {self.file_path}")
        if self.file_path and os.path.exists(self.file_path):
            return self.load_from_excel()
        elif self.symbol:
            return self.load_from_yfinance()
        else:
            raise ValueError("No valid data source provided.")

    def load_from_excel(self):
        try:
            self.df = pd.read_excel(self.file_path)
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            close_columns = [col for col in self.df.columns if col.lower() == 'close']
            if not close_columns:
                raise ValueError("No 'Close' column found in the Excel file.")
            return self.df[close_columns[0]].tolist()
        except Exception as e:
            print(f"Error reading Excel file: {e}")
            raise

    def load_from_yfinance(self):
        try:
            self.df = yf.download(self.symbol, start=self.start_date, end=self.end_date, interval='1h')
            if self.df.empty:
                raise ValueError("Yahoo Finance returned no data.")
            return self.df['Close'].tolist()
        except Exception as e:
            raise ValueError(f"Error fetching data from Yahoo Finance: {e}")

    # Griffiths Methods
    def highpass_filter(self, price_series, period):
        if len(price_series) < 3:
            return [0.0] * len(price_series)
        a1 = math.exp(-1.414 * math.pi / period)
        b1 = 2 * a1 * math.cos(1.414 * 180 / period)
        c1, c2, c3 = (1 + b1) / 4, b1, -(a1 ** 2)
        hp_series = [0.0, 0.0]
        for i in range(2, len(price_series)):
            hp_series.append(c1 * (price_series[i] - 2 * price_series[i - 1] + price_series[i - 2]) + 
                            c2 * hp_series[i - 1] + c3 * hp_series[i - 2])
        return hp_series

    def super_smoother(self, price_series, period):
        if len(price_series) < 3:
            return price_series
        a1 = math.exp(-1.414 * math.pi / period)
        b1 = 2 * a1 * math.cos(1.414 * 180 / period)
        c1, c2, c3 = 1 - b1 + a1**2, b1, -(a1**2)
        ss_series = [price_series[0], price_series[1]]
        for i in range(2, len(price_series)):
            ss_series.append(c1 * (price_series[i] + price_series[i - 1]) / 2 + 
                            c2 * ss_series[i - 1] + c3 * ss_series[i - 2])
        return ss_series

    def detect_cycles(self, close_prices):
        hp = self.highpass_filter(close_prices, self.griffiths_upper_bound)
        lp = self.super_smoother(hp, self.griffiths_lower_bound)
        dominant_cycles = []
        for i in range(len(lp) - self.griffiths_length):
            segment = lp[i : i + self.griffiths_length]
            max_power, dominant_cycle = 0, 0
            for period in range(self.griffiths_lower_bound, self.griffiths_upper_bound + 1):
                real_part = sum(segment[j] * math.cos(2 * math.pi * j / period) for j in range(self.griffiths_length))
                imag_part = sum(segment[j] * math.sin(2 * math.pi * j / period) for j in range(self.griffiths_length))
                power = 0.1 / ((1 - real_part) ** 2 + imag_part ** 2) if (1 - real_part) ** 2 + imag_part ** 2 != 0 else 0
                if power > max_power:
                    max_power, dominant_cycle = power, period
            dominant_cycles.append(dominant_cycle)
        return dominant_cycles

    def griffiths_trend(self, dominant_cycles):
        trend_labels = []
        for i in range(len(dominant_cycles) - self.griffiths_window):
            window_cycles = dominant_cycles[i : i + self.griffiths_window]
            cycle_variability = max(window_cycles) - min(window_cycles)
            trend_labels.append('Trending' if cycle_variability > self.griffiths_stability_threshold else 'Non-Trending')
        return trend_labels

    # USI Method
    def usi_trend(self, close_prices):
        df = pd.DataFrame({'Close': close_prices})
        usi_values = calculate_usi(df=df, length=self.usi_length, window=self.usi_window)
        trend_labels = []
        for usi in usi_values:
            if pd.isna(usi):
                trend_labels.append('Non-Trending')
            else:
                trend_labels.append('Trending' if abs(usi) > self.usi_trend_threshold else 'Non-Trending')
        return trend_labels

    def combine_trends(self, usi_labels, griffiths_labels):
        min_length = min(len(usi_labels), len(griffiths_labels))
        usi_labels = usi_labels[:min_length]
        griffiths_labels = griffiths_labels[:min_length]
        combined_labels = []
        for u, g in zip(usi_labels, griffiths_labels):
            if u == 'Trending' or g == 'Trending':  # "OR" rule; change to "and" for stricter
                combined_labels.append('Trending')
            else:
                combined_labels.append('Non-Trending')
        return combined_labels

    def run(self):
        close_prices = self.fetch_data()
        if not close_prices:
            return "No data available."

        usi_labels = self.usi_trend(close_prices)
        griffiths_cycles = self.detect_cycles(close_prices)
        griffiths_labels = self.griffiths_trend(griffiths_cycles)
        trend_classification = self.combine_trends(usi_labels, griffiths_labels)

        return self.analyze_trend_periods(trend_classification)

    def analyze_trend_periods(self, trend_labels):
        dates = self.df['Date'].tolist()
        if len(dates) != len(trend_labels):
            min_length = min(len(dates), len(trend_labels))
            dates = dates[:min_length]
            trend_labels = trend_labels[:min_length]

        trend_periods = []
        current_trend = trend_labels[0]
        current_start = dates[0]

        for i in range(1, len(trend_labels)):
            if trend_labels[i] != current_trend:
                trend_periods.append({
                    'trend': current_trend,
                    'start_date': current_start,
                    'end_date': dates[i-1],
                    'duration': (dates[i-1] - current_start).total_seconds() / 3600  # Hours
                })
                current_trend = trend_labels[i]
                current_start = dates[i]

        trend_periods.append({
            'trend': current_trend,
            'start_date': current_start,
            'end_date': dates[-1],
            'duration': (dates[-1] - current_start).total_seconds() / 3600  # Hours
        })

        return trend_periods

if __name__ == "__main__":
    try:
        detector = TrendDetector(
            symbol='SPY',
            start_date=datetime.now() - timedelta(days=730),
            end_date=datetime.now(),
            usi_length=28,
            usi_window=4,
            usi_trend_threshold=0.5,
            griffiths_lower_bound=18,
            griffiths_upper_bound=40,
            griffiths_length=40,
            griffiths_window=10,
            griffiths_stability_threshold=5
        )
        trend_periods = detector.run()

        print("\nUSI and Griffiths Combined Trend Analysis:")
        for period in trend_periods:
            print(f"Trend: {period['trend']}")
            print(f"  Start Date: {period['start_date']}")
            print(f"  End Date: {period['end_date']}")
            print(f"  Duration: {period['duration']:.2f} hours")
            print()
    except Exception as e:
        print(f"Error: {e}")