import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

from src.Filters.high_pass_2pole_filter import highpass_2pole_filter
from src.Filters.low_pass_2pole_filter import super_smoother


class GriffithsPredictor:
    def __init__(
        self,
        close_prices,  # raw close price series
        make_stationary: bool = True,
        use_log_diff: bool = False,
        length: int = 18,
        lower_bound: int = 18,
        upper_bound: int = 40,
        bars_fwd: int = 2,
        peak_decay: float = 0.991,
        initial_peak: float = None,
    ):
        """
        Initializes the Griffiths Predictor.

        Args:
            close_prices (array-like): The raw close price series.
            make_stationary (bool): If True, the predictor uses a stationary version of the data.
                If False, it uses the original data.
            use_log_diff (bool): If True and make_stationary is True, use log differences
                (i.e. log returns) instead of percent changes.
            length (int): The window length for prediction.
            lower_bound (int): Parameter for the low-pass (super smoother) filter.
            upper_bound (int): Parameter for the high-pass filter.
            bars_fwd (int): Number of future periods to forecast.
            peak_decay (float): Decay factor applied to the running peak.
            initial_peak (float, optional): Initial peak value. If None, computed from the input series.
        """
        self.close_prices = np.asarray(close_prices)
        self.make_stationary = make_stationary
        self.use_log_diff = use_log_diff
        self.length = length
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.bars_fwd = bars_fwd
        self.peak_decay = peak_decay        
        
        # Convert close_prices to a numpy array and squeeze to ensure a 1D array.
        self.close_prices = np.asarray(close_prices).squeeze()
        if self.close_prices.ndim != 1:
            raise ValueError("close_prices must be a 1D array after squeezing.")
        
        # Define the input series based on the flags.
        if self.make_stationary:
            self.input_series = np.zeros_like(self.close_prices, dtype=float)
            if len(self.close_prices) > 1:
                if self.use_log_diff:
                    # Use log differences: log(close[t]) - log(close[t-1])
                    self.input_series[1:] = np.diff(np.log(self.close_prices))
                    self.stationary_type = "Log Difference"
                else:
                    # Use daily percent change.
                    self.input_series[1:] = (self.close_prices[1:] - self.close_prices[:-1]) / self.close_prices[:-1]
                    self.stationary_type = "Percent Change"
            else:
                self.input_series[0] = 0.0
        else:
            self.input_series =self.close_prices.copy()

        # Compute an initial peak if not provided.
        if initial_peak is None:
            self.initial_peak = self.compute_initial_peak(self.input_series)
        else:
            self.initial_peak = initial_peak

    @staticmethod
    def compute_initial_peak(series, window: int = 30) -> float:
        """
        Computes an initial peak value based on historical volatility.
        For a percent change series, this is the maximum absolute value over the past `window` observations.

        Args:
            series (array-like): The input series.
            window (int): Look-back window.
        
        Returns:
            float: The computed initial peak.
        """
        s = np.asarray(series)
        init_peak = 0.01
        if len(s) != 0:
            init_peak = np.max(np.abs(s[-window:])) if len(s) >= window else np.max(np.abs(s))
        
        print(f"Initial peak = {init_peak}")
        return init_peak

    def _predict(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Runs the core predictor on self.input_series.

        Returns:
            tuple: (predictions, future_signals) as arrays in the same space as self.input_series.
        """
        mu = 1 / self.length  # Learning rate

        # Apply filters to the input series.
        hp = highpass_2pole_filter(self.input_series, self.upper_bound)
        bp = super_smoother(hp, self.lower_bound)  # bandpass filter

        # Initialize sliding window and coefficient vector.
        xx = np.zeros(self.length)
        coef = np.zeros(self.length)
        peak = self.initial_peak
        predictions = np.zeros_like(self.input_series)

        # Process the filtered signal starting from index 'length'
        for t in range(self.length, len(bp)):
            peak = self.peak_decay * peak
            if np.abs(bp[t]) > peak:
                peak = np.abs(bp[t])
            signal_val = bp[t] / peak if peak != 0 else 0

            # Shift window and insert new value.
            xx[:-1] = xx[1:]
            xx[-1] = signal_val

            # Use reversed window for prediction.
            reversed_xx = xx[::-1]
            prediction = np.dot(reversed_xx, coef)
            predictions[t] = prediction

            error = reversed_xx[0] - prediction
            coef += mu * error * reversed_xx

        # Forecast future values.
        future_signals = np.zeros(self.bars_fwd)
        for i in range(self.bars_fwd):
            reversed_xx = xx[::-1]
            future_signal = np.dot(reversed_xx, coef)
            future_signals[i] = future_signal
            xx[:-1] = xx[1:]
            xx[-1] = future_signal

        return predictions, future_signals

    def predict_stationary(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Runs the predictor and returns predictions in percent change space.
        This method is only valid if make_stationary is True.
        
        Returns:
            tuple: (pct_predictions, future_pct) as arrays.
        """
        if not self.make_stationary:
            raise ValueError("predict_pct_change is only applicable when make_stationary is True.")
        return self._predict()

    def predict_price(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Runs the predictor and returns a forecast in price space.
        
        If make_stationary is True, it uses the percent change predictions to forecast future prices,
        compounding from the last observed close price.
        If make_stationary is False, the predictions are assumed to be in price space already.
        
        Returns:
            tuple: (historical_prediction, future_price) as arrays.
        """
        predictions, future_signals = self._predict()
        if self.make_stationary:
            # Convert percent change predictions to price forecasts.
            last_price = self.close_prices[-1]
            # Compute future price forecast from predicted percent changes.
            future_price = np.empty_like(future_signals)
            if len(future_signals) > 0:
                future_price[0] = last_price * (1 + future_signals[0])
                for i in range(1, len(future_signals)):
                    future_price[i] = future_price[i - 1] * (1 + future_signals[i])
            # Optionally, also convert historical percent change predictions to price predictions.
            # Here, we create a historical predicted price series from one-step forecasts.
            historical_price_pred = np.full_like(self.close_prices, np.nan, dtype=float)
            for t in range(self.length, len(self.close_prices)):
                # One-step forecast: use previous close price.
                historical_price_pred[t] = self.close_prices[t - 1] * (1 + predictions[t])
            return historical_price_pred, future_price
        else:
            # If not stationary, predictions are already in price space.
            return predictions, future_signals

