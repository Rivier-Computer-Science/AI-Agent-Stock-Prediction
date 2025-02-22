import numpy as np
import pandas as pd
import math
import yfinance as yf
import plotly.graph_objects as go
import math


def to_float_list(data):

    if isinstance(data, pd.DataFrame):
        if data.shape[1] == 1:
            data = data.squeeze("columns")  
        else:
            raise ValueError(
                "DataFrame has multiple columns. Please pass a single-column DataFrame or a Series."
            )


    if isinstance(data, pd.Series):
        data = data.reset_index(drop=True)
        data = data.tolist()

    if isinstance(data, np.ndarray):
        if data.ndim == 2:
            raise ValueError("Data appears to be 2D, expected 1D array for time series.")
        data = data.tolist()

    if not isinstance(data, list):
        raise ValueError("Data could not be converted to a Python list.")

    if any(isinstance(x, (list, np.ndarray)) for x in data):
        raise ValueError(
            "Data appears to be multi-dimensional (list of lists). Expected 1D list of numeric values."
        )

    data = [float(x) for x in data]
    return data

def highpass_filter(price_series, period):

    price_series = to_float_list(price_series)

    if len(price_series) < 3:
        return [0.0] * len(price_series)

    a1 = math.exp(-1.414 * math.pi / period)
    b1 = 2 * a1 * math.cos(1.414 * 180 / period)
    c1 = (1 + b1) / 4
    c2 = b1
    c3 = -(a1 ** 2)

    highpass_series = [0.0, 0.0]  

    for i in range(2, len(price_series)):
        hp_value = (
            c1 * (price_series[i] - 2 * price_series[i - 1] + price_series[i - 2])
            + c2 * highpass_series[i - 1]
            + c3 * highpass_series[i - 2]
        )
        highpass_series.append(hp_value)

    return highpass_series

def super_smoother(price_series, period):

    price_series = to_float_list(price_series)

    if len(price_series) < 2:
        return price_series

    a1 = math.exp(-1.414 * math.pi / period)
    b1 = 2 * a1 * math.cos(1.414 * 180 / period)
    c1 = 1 - b1 + a1**2
    c2 = b1
    c3 = -(a1**2)

    ss = [0.0] * len(price_series)
    ss[0] = price_series[0]
    ss[1] = price_series[1]

    for i in range(2, len(price_series)):
        ss[i] = (
            c1 * (price_series[i] + price_series[i - 1]) / 2
            + c2 * ss[i - 1]
            + c3 * ss[i - 2]
        )
    return ss

class CycleDetector:
    def __init__(self, symbol, start_date, end_date,
                 lower_bound=18, upper_bound=40, length=40):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.length = length
        self.mu = 1 / self.length

    def fetch_data(self):

        df = yf.download(self.symbol, start=self.start_date, end=self.end_date)
        if df.empty:
            return []
        
        close_series = df[['Close']].squeeze("columns")
        return to_float_list(close_series)

    def calculate_cycles(self, close_prices):

        close_prices = to_float_list(close_prices)
        if len(close_prices) < self.length:
            raise ValueError(
                f"Not enough data to compute cycles with length={self.length}. "
                f"Got {len(close_prices)} data points."
            )

        hp = highpass_filter(close_prices, self.upper_bound)
        lp = super_smoother(hp, self.lower_bound)
        lp_arr = np.array(lp, dtype=float)

        peak = 0.1
        current_peak = np.max(np.abs(lp_arr))
        if current_peak > peak:
            peak = current_peak
        if peak != 0:
            signal = lp_arr / peak
        else:
            signal = np.zeros_like(lp_arr)

        coefficients = np.zeros(self.length)
        xx = np.zeros(self.length)


        for i in range(len(signal)):
            xx[1:] = xx[:-1]
            xx[0] = signal[i]

            if i >= self.length - 1:
                x_bar = np.dot(xx, coefficients)
                coefficients += self.mu * (xx[0] - x_bar) * xx

        power = np.zeros((self.upper_bound + 1, 2))
        max_power = 0.0
        dominant_cycle = 0

        for period in range(self.lower_bound, self.upper_bound + 1):
            real_part = np.sum(coefficients * np.cos(2 * np.pi * np.arange(1, self.length + 1) / period))
            imag_part = np.sum(coefficients * np.sin(2 * np.pi * np.arange(1, self.length + 1) / period))
            denom = (1 - real_part)**2 + imag_part**2
            power_val = 0.1 / denom if denom != 0 else 0
            power[period, 1] = power_val

            if power_val > max_power:
                max_power = power_val
                dominant_cycle = period

        return {
            "dominant_cycle": dominant_cycle,
            "close_prices": close_prices,
            "hp": hp,
            "lp": lp
        }

"""
# Example usage:
detector = CycleDetector('ES=F', '2024-01-01', '2024-12-31')
data = detector.fetch_data()
if len(data) == 0:
    print("No data available or data is empty.")
else:
    results = detector.calculate_cycles(data)

    dominant_cycle = results["dominant_cycle"]
    close_prices = results["close_prices"]
    hp = results["hp"]
    lp = results["lp"]

    print(f"Dominant cycle length: {dominant_cycle}")

    x_vals = list(range(len(close_prices)))
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=close_prices,
            mode='lines',
            name='Close'
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=hp,
            mode='lines',
            name='Highpass Filter'
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=lp,
            mode='lines',
            name='SuperSmoother'
        )
    )

    if dominant_cycle > 0:
        sin_wave = [
            math.sin(2 * math.pi * (i / dominant_cycle)) for i in x_vals
        ]
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=sin_wave,
                mode='lines',
                name=f'Sine wave (period={dominant_cycle})'
            )
        )

    fig.update_layout(
        title=f'Dominant Cycle = {dominant_cycle}',
        xaxis_title='Index (Time)',
        yaxis_title='Value',
        legend_title='Signals'
    )

    fig.show()
"""