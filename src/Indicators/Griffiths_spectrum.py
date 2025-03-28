import math
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.Indicators.High_pass_filter_function import highpass_filter
from src.Indicators.SuperSmoother_filter_function import super_smoother

def griffiths_spectrum(
    symbol="AAPL",
    start_date="2020-01-01",
    end_date="2020-12-31",
    hp_period=30,
    ss_period=14,
    lower_bound=10,
    upper_bound=40,
    length=40
):
    # 1) Fetch data
    df = yf.download(symbol, start=start_date, end=end_date)
    if df.empty:
        print("No data returned for the given range.")
        return

    # Handle possibility of df["Close"] being DataFrame
    df_close = df["Close"]
    if isinstance(df_close, pd.DataFrame):
        if df_close.shape[1] == 1:
            df_close = df_close.squeeze("columns")
        else:
            raise ValueError("Multiple columns in df['Close']. Handle separately.")

    close_prices = df_close.reset_index(drop=True).tolist()
    n_bars = len(close_prices)

    # 2) Compute HP & SS across entire series
    hp_series = highpass_filter(close_prices, hp_period)
    ss_series = super_smoother(hp_series, ss_period)

    # 3) Build the Griffiths Spectrum as a Heatmap (bar-by-bar approach, per Ehlers code)
    pwr_array = np.zeros((n_bars, upper_bound + 1))  # shape (time, period)
    coef = np.zeros(length)
    XX = np.zeros(length)
    peak = 0.1
    signals = np.zeros(n_bars)

    for t in range(n_bars):
        # Decay the previous peak
        peak *= 0.991
        # Update peak if needed
        if abs(ss_series[t]) > peak:
            peak = abs(ss_series[t])

        # Compute normalized signal
        if peak != 0:
            signals[t] = ss_series[t] / peak
        else:
            signals[t] = 0

        # Fill XX[...] with last 'length' signals
        window_start = max(0, t - length + 1)
        window_vals = signals[window_start : t + 1]
        XX[:] = 0
        for i, val in enumerate(reversed(window_vals)):
            if i < length:
                XX[i] = val

        # Update Coefficients once we have enough bars
        if t >= length - 1:
            x_bar = 0.0
            for c in range(1, length + 1):
                x_bar += XX[length - c] * coef[c - 1]

            correction = XX[length - 1] - x_bar
            for c in range(1, length + 1):
                coef[c - 1] += (1.0 / length) * correction * XX[length - c]

        # Now compute power for each period
        for period in range(lower_bound, upper_bound + 1):
            real_val = 0.0
            imag_val = 0.0
            for c in range(1, length + 1):
                angle = math.radians(360 * c / period)
                real_val += coef[c - 1] * math.cos(angle)
                imag_val += coef[c - 1] * math.sin(angle)

            denom = (1.0 - real_val)**2 + (imag_val**2)
            old_pwr = pwr_array[t - 1, period] if t > 0 else 0.0
            new_pwr = 0.0
            if denom != 0:
                new_pwr = 0.1 / denom + 0.9 * old_pwr

            pwr_array[t, period] = new_pwr

        # Normalize the row so max = 1.0
        row_max = pwr_array[t, lower_bound : (upper_bound + 1)].max()
        if row_max > 0:
            pwr_array[t, lower_bound : (upper_bound + 1)] /= row_max

    # 4) Plot with THREE subplots
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.07,
        subplot_titles=(
            "Original Close",
            "Highpass & SuperSmoother",
            "Griffiths Spectrum Heatmap"
        )
    )

    # ---- 1) TOP ROW: Original Close
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=close_prices,
            mode='lines',
            name='Close (original)',
        ),
        row=1, col=1
    )

    # ---- 2) MIDDLE ROW: HP & SS
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=hp_series,
            mode='lines',
            name=f'Highpass (P={hp_period})'
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=ss_series,
            mode='lines',
            name=f'SuperSmoother (P={ss_period})'
        ),
        row=2, col=1
    )

    # ---- 3) BOTTOM ROW: Heatmap
    x_values = df.index  # or just range(n_bars) if you prefer
    y_values = list(range(lower_bound, upper_bound + 1))
    z_matrix = pwr_array[:, lower_bound : (upper_bound + 1)]
    z_matrix_T = z_matrix.T  # shape (#periods, n_bars)

    heatmap = go.Heatmap(
        x=x_values,
        y=y_values,
        z=z_matrix_T,
        colorscale='Viridis',
        name="Griffiths Spectrum"
    )
    fig.add_trace(heatmap, row=3, col=1)

    # Layout updates
    fig.update_layout(
        title=f"Filters + Griffiths Spectrum Heatmap: {symbol} ({start_date} to {end_date})",
        height=900,
        showlegend=True
    )

    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="HP/SS", row=2, col=1)

    fig.update_xaxes(title_text="Time (Bars)", row=3, col=1)
    fig.update_yaxes(title_text="Cycle Period", row=3, col=1)

    fig.show()


# Example usage
if __name__ == "__main__":
    griffiths_spectrum(
        symbol="ES=F",
        start_date="2023-09-01",
        end_date="2024-08-31",
        hp_period=30,
        ss_period=14,
        lower_bound=10,
        upper_bound=40,
        length=40
    )
