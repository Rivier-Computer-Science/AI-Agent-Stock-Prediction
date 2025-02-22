import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

from src.Filters.high_pass_2pole_filter import highpass_2pole_filter
from src.Filters.low_pass_2pole_filter import super_smoother


class GriffithsPredictor:
    def __init__(
        self,
        close_prices,  # absolute close price series
        length: int = 18,
        lower_bound: int = 18,
        upper_bound: int = 40,
        bars_fwd: int = 2,
        peak_decay: float = 0.991,
        initial_peak: float = None,
    ):
        """
        Initializes the Griffiths Predictor with close price data.
        The constructor computes the daily percent change series from the provided
        close prices and, if not provided, computes an initial peak from the percent change data.

        Args:
            close_prices (array-like): The absolute close price series.
            length (int): The window length for prediction.
            lower_bound (int): Parameter for the low-pass (super smoother) filter.
            upper_bound (int): Parameter for the high-pass filter.
            bars_fwd (int): Number of future periods to forecast.
            peak_decay (float): Decay factor applied to the running peak.
            initial_peak (float, optional): Initial peak value. If None, computed from percent change data.
        """
        self.close_prices = np.asarray(close_prices)
        self.length = length
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.bars_fwd = bars_fwd
        self.peak_decay = peak_decay

        # Compute daily percent change.
        # We'll define the percent change series with the same length as close_prices,
        # setting the first element to 0 (or you could drop it later).
        self.pct_series = np.zeros_like(self.close_prices, dtype=float)
        if len(self.close_prices) > 1:
            self.pct_series[1:] = (self.close_prices[1:] - self.close_prices[:-1]) / self.close_prices[:-1]
        else:
            self.pct_series[0] = 0.0

        # Compute initial peak from the percent change series if not provided.
        if initial_peak is None:
            self.initial_peak = self.compute_initial_peak(self.pct_series[1:])  # ignore the first zero
        else:
            self.initial_peak = initial_peak

    @staticmethod
    def compute_initial_peak(price_series, window: int = 30) -> float:
        """
        Computes an initial peak value based on historical volatility.
        Since the input is the percent change series, this computes the maximum
        absolute value over the last `window` observations.
        
        Args:
            price_series (array-like): The percent change series.
            window (int): Look-back window.
        
        Returns:
            float: The computed initial peak.
        """
        series = np.asarray(price_series)
        if len(series) == 0:
            return 0.01
        return np.max(np.abs(series[-window:])) if len(series) >= window else np.max(np.abs(series))

    def predict_pct_change(self) -> (np.ndarray, np.ndarray):
        """
        Runs the Griffiths Predictor on the percent change series.
        The process:
          1. Applies a high-pass filter then a low-pass (super smoother) filter.
          2. Maintains a sliding window of normalized signal values.
          3. Uses an adaptive linear model to predict the signal.
          4. Forecasts future values for the specified number of bars.
        
        Returns:
            tuple: (predictions, future_signals) as NumPy arrays of percent changes.
        """
        mu = 1 / self.length  # learning rate
        # Apply filters to the percent change series.
        hp = highpass_2pole_filter(self.pct_series, self.upper_bound)
        lp = super_smoother(hp, self.lower_bound)

        # Initialize sliding window (xx) and predictor coefficients.
        xx = np.zeros(self.length)
        coef = np.zeros(self.length)
        peak = self.initial_peak
        predictions = np.zeros_like(self.pct_series)

        # Process filtered signal starting from index "length".
        for t in range(self.length, len(lp)):
            peak = self.peak_decay * peak
            if np.abs(lp[t]) > peak:
                peak = np.abs(lp[t])
            signal_val = lp[t] / peak if peak != 0 else 0

            # Update sliding window: shift left and add new value.
            xx[:-1] = xx[1:]
            xx[-1] = signal_val

            # Use reversed window to mimic original order.
            reversed_xx = xx[::-1]
            prediction = np.dot(reversed_xx, coef)
            predictions[t] = prediction

            error = reversed_xx[0] - prediction
            coef += mu * error * reversed_xx

        # Forecast future percent changes.
        future_signals = np.zeros(self.bars_fwd)
        for i in range(self.bars_fwd):
            reversed_xx = xx[::-1]
            future_signal = np.dot(reversed_xx, coef)
            future_signals[i] = future_signal
            xx[:-1] = xx[1:]
            xx[-1] = future_signal

        return predictions, future_signals

    def predict_price(self) -> (np.ndarray, np.ndarray):
        """
        Runs the predictor on the percent change series, then converts the future forecast
        from percent change to price predictions using the last observed close price.
        
        Returns:
            tuple: (historical_pct_predictions, future_price_forecast)
                   - historical_pct_predictions is an array of predicted percent changes (same as predict_pct_change)
                   - future_price_forecast is an array of predicted future prices.
        """
        pct_predictions, pct_future = self.predict_pct_change()

        # For forecasting prices, use the last observed close price.
        last_price = self.close_prices[-1]
        future_price = np.empty_like(pct_future)
        if len(pct_future) > 0:
            # The first future price is last_price adjusted by the first predicted percent change.
            future_price[0] = last_price * (1 + pct_future[0])
            # Subsequent future prices are computed cumulatively.
            for i in range(1, len(pct_future)):
                future_price[i] = future_price[i-1] * (1 + pct_future[i])
        else:
            future_price = np.array([])

        return pct_predictions, future_price

