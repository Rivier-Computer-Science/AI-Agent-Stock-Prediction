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
        scale_to_price: bool = False
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
        self.scale_to_price = scale_to_price 
        
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
        Runs the Griffiths predictor logic on self.input_series
        and optionally denormalizes the results to approximate
        a 'price-like' scale.

        Returns:
            tuple: (predictions, future_signals)
                If denormalize=False, both arrays are in the same
                normalized domain as Ehlers's "Signal".
                If denormalize=True, they represent an approximate
                'LP' price in real-time for historical bars, and a
                forward forecast (in 'LP' scale) for future bars.
        """
        mu = 1 / self.length  # Learning rate

        # ------------------------------------------------------
        # 1) EHLERS FILTERS: HP & LP
        # ------------------------------------------------------
        # Because Ehlers code does:
        #    HP = $HighPass(Close, UpperBound)
        #    LP = $SuperSmoother(HP, LowerBound)
        hp = highpass_2pole_filter(self.input_series, self.upper_bound)
        lp = super_smoother(hp, self.lower_bound)

        # ------------------------------------------------------
        # 2) CREATE THE 'signal' ARRAY WITH PEAK TRACKING
        # ------------------------------------------------------
        # In Ehlers code:
        #    Peak = .991 * Peak[1]
        #    If AbsValue(LP) > Peak Then Peak = AbsValue(LP)
        #    Signal = LP / Peak (if Peak != 0)
        # We'll do that bar-by-bar.
        n = len(lp)
        signal = np.zeros(n)
        peak_array = np.zeros(n)

        peak = self.initial_peak
        for t in range(n):
            # Exponential decay of old peak
            peak *= self.peak_decay

            # Immediate reset if new amplitude is bigger
            if abs(lp[t]) > peak:
                peak = abs(lp[t])

            peak_array[t] = peak
            signal[t] = lp[t] / peak if peak != 0.0 else 0.0

        # ------------------------------------------------------
        # 3) PRELOAD THE SLIDING WINDOW 'xx' FROM THE LAST 'length' BARS
        # ------------------------------------------------------
        xx = np.zeros(self.length)
        coef = np.zeros(self.length)
        
        # Fill xx so that xx[0] is the oldest among those 'length' bars
        # and xx[-1] is the newest.
        # Exactly like Ehlers snippet:
        #   For count=1 to Length:
        #       XX[count] = Signal[Length - count];
        # except we do it in Python indexing:
        preload_start = max(0, len(signal) - self.length)  # in case length > data
        initial_chunk = signal[preload_start : preload_start + self.length]
        xx[:] = initial_chunk  # oldest in xx[0], newest in xx[-1]

        # We'll store predictions for each bar in the same array length
        predictions = np.zeros_like(signal)
        # Because we haven't adapted yet for bars < self.length,
        # we can just leave them zero or NaN. We'll do zero for simplicity.

        # ------------------------------------------------------
        # 4) MAIN LOOP: LMS ADAPTATION
        # ------------------------------------------------------
        # We'll adapt from t = length up to the end. (Ehlers code does
        # its array indexing differently, but conceptually it starts
        # updating once we have length samples.)
        for t in range(self.length, len(signal)):
            # The newest normalized sample is signal[t].
            # Shift 'xx' left by 1
            xx[:-1] = xx[1:]
            # Insert the new sample at the end
            xx[-1] = signal[t]

            # Reverse so reversed_xx[0] is the newest
            reversed_xx = xx[::-1]

            # LMS prediction
            prediction = np.dot(reversed_xx, coef)
            predictions[t] = prediction

            # Error = newest sample (reversed_xx[0]) - predicted
            error = reversed_xx[0] - prediction
            # Coeff update
            coef += mu * error * reversed_xx

        # ------------------------------------------------------
        # 5) FORECAST FUTURE VALUES (still in normalized domain)
        # ------------------------------------------------------
        future_signals = np.zeros(self.bars_fwd)
        for i in range(self.bars_fwd):
            reversed_xx = xx[::-1]
            future_signal = np.dot(reversed_xx, coef)
            future_signals[i] = future_signal

            # Shift left, add forecast to end
            xx[:-1] = xx[1:]
            xx[-1] = future_signal

        # --------------------------------------------------
        # 6) SCALE BACK TO PRICE USING LINEAR APPROXIMATION
        # -------------------------------------------------
        if self.scale_to_price:
            predictions, future_signals = self.scale_predictions_to_price(self.close_prices, predictions, future_signals)

        return predictions, future_signals


    def scale_predictions_to_price(self,
                                price_hist: np.ndarray,
                                predictions: np.ndarray,
                                future_signals: np.ndarray
                                ) -> tuple[np.ndarray, np.ndarray]:
        """
        Post-process dimensionless Ehlers predictions to approximate
        the scale of 'price_hist', ignoring the first self.length bars
        where predictions are zero/uninitialized.

        Steps:
        1) Exclude [0..self.length-1] from the scaling calculation.
        2) Compute a linear mapping a + b*X so the valid predictions
        match the mean/std of the valid price.
        3) Apply to all in-sample predictions and future signals.
        4) Prepend the first self.length in-sample values as zero
        to maintain date alignment.

        Args:
            price_hist:      The historical price array, shape (N,).
            predictions:     Dimensionless in-sample predictions, shape (N,).
            future_signals:  Dimensionless out-of-sample predictions, shape (F,).

        Returns:
            (scaled_in_sample, scaled_future):
                - scaled_in_sample:  shape (N,), in approximate "price" domain.
                - scaled_future:     shape (F,), forward forecast in the same scale.
        """

        # 1) Slice off the initial warmup region
        warmup = self.length
        valid_price = price_hist[warmup:]
        valid_pred  = predictions[warmup:]

        # Edge check
        if len(valid_price) == 0 or len(valid_pred) == 0:
            # If self.length >= len(price_hist), fallback
            return predictions, future_signals

        # 2) Compute mean & std for the valid region
        mean_price = np.mean(valid_price)
        mean_pred  = np.mean(valid_pred)
        std_price  = np.std(valid_price)
        std_pred   = np.std(valid_pred)

        # Avoid divide-by-zero
        if std_pred < 1e-12:
            std_pred = 1.0

        # 3) Determine the linear transform: scaled = a + b*pred
        b = std_price / std_pred
        a = mean_price - b * mean_pred

        # 4) Apply the transform to the valid portion of in-sample
        scaled_valid_pred = a + b * valid_pred

        # 5) Build the final scaled_in_sample array
        #    The first self.length bars remain zero (or whatever they were).
        scaled_in_sample = np.zeros_like(predictions)
        scaled_in_sample[:warmup] = 0.0
        scaled_in_sample[warmup:] = scaled_valid_pred

        # 6) Scale the out-of-sample predictions
        scaled_future = a + b * future_signals

        return scaled_in_sample, scaled_future





