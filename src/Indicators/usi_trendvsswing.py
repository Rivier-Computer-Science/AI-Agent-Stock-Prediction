import numpy as np
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

# Simplified USI Calculation
def calculate_usi(prices, period, smoothing_period=4):
    """
    Simplified USI calculation with vectorized operations, handling 1D input
    """
    # Ensure prices is 1D
    prices = np.asarray(prices).flatten()
    
    # Vectorized SU/SD calculation
    diff = np.diff(prices)
    su = np.where(diff > 0, diff, 0)
    sd = np.where(diff < 0, -diff, 0)
    
    # Pad with zeros to match original length, ensuring 1D arrays
    su = np.concatenate((np.array([0]), su))
    sd = np.concatenate((np.array([0]), sd))
    
    # Smoothing with rolling mean and UltimateSmoother
    su_avg = pd.Series(su).rolling(window=smoothing_period, min_periods=1).mean().values
    sd_avg = pd.Series(sd).rolling(window=smoothing_period, min_periods=1).mean().values
    
    usu = ultimate_smoother(su_avg, period)
    usd = ultimate_smoother(sd_avg, period)
    
    # Vectorized USI computation with safety checks
    denominator = usu + usd
    mask = (denominator != 0) & (usu > 0.01) & (usd > 0.01)
    usi = np.where(mask, (usu - usd) / denominator, 0)
    
    return usi

def ultimate_smoother(series, period):
    """
    Optimized UltimateSmoother using pandas rolling
    """
    series = pd.Series(series)
    a1 = np.exp(-1.414 * np.pi / period)
    b1 = 2 * a1 * np.cos(1.414 * np.pi / period)
    c2 = b1
    c3 = -a1 * a1
    c1 = (1 + c2 - c3) / 4
    
    # Initialize output
    us = series.copy()
    
    # Apply filter for bars >= 4
    for i in range(4, len(series)):
        us.iloc[i] = (1 - c1) * series.iloc[i] + \
                    (2 * c1 - c2) * series.iloc[i-1] - \
                    (c1 + c3) * series.iloc[i-2] + \
                    c2 * us.iloc[i-1] + \
                    c3 * us.iloc[i-2]
    
    return us.values

# Fetch real-world data
def get_real_data(ticker="SPY", start="2023-01-01", end="2024-06-30"):
    data = yf.download(ticker, start=start, end=end, interval="1d")
    return data['Close'].values, data.index

# Visualization Functions for Task 2
def plot_usi_with_features(prices, dates, period, title):
    usi = calculate_usi(prices, period)
    
    # Detect zero crossings
    zero_crossings_up = np.where((usi[:-1] < 0) & (usi[1:] >= 0))[0] + 1
    zero_crossings_down = np.where((usi[:-1] > 0) & (usi[1:] <= 0))[0] + 1
    
    # Trend line (20-bar SMA)
    usi_sma = pd.Series(usi).rolling(window=20, min_periods=1).mean().values
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=usi, name=f'USI ({period} bars)', line=dict(color='blue')))
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    fig.add_trace(go.Scatter(x=[dates[i] for i in zero_crossings_up], 
                            y=[usi[i] for i in zero_crossings_up], 
                            mode='markers', name='Bullish Cross', 
                            marker=dict(color='green', size=10, symbol='triangle-up')))
    fig.add_trace(go.Scatter(x=[dates[i] for i in zero_crossings_down], 
                            y=[usi[i] for i in zero_crossings_down], 
                            mode='markers', name='Bearish Cross', 
                            marker=dict(color='red', size=10, symbol='triangle-down')))
    fig.add_trace(go.Scatter(x=dates, y=usi_sma, name='Trend (20-bar SMA)', 
                            line=dict(color='orange', dash='dot')))
    
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title='USI Value', 
                     template='plotly_white', height=400)
    return fig

def plot_side_by_side_comparison():
    prices, dates = get_real_data()
    
    fig = make_subplots(rows=2, cols=1, 
                       subplot_titles=('USI (112 bars) - Trend Trading', 'USI (28 bars) - Swing Trading'),
                       vertical_spacing=0.1)
    
    # Trend Trading (112 bars)
    usi_trend = calculate_usi(prices, period=112)
    zero_crossings_up_trend = np.where((usi_trend[:-1] < 0) & (usi_trend[1:] >= 0))[0] + 1
    zero_crossings_down_trend = np.where((usi_trend[:-1] > 0) & (usi_trend[1:] <= 0))[0] + 1
    usi_sma_trend = pd.Series(usi_trend).rolling(window=20, min_periods=1).mean().values
    
    fig.add_trace(go.Scatter(x=dates, y=usi_trend, name='USI (112 bars)', line=dict(color='blue')), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=1)
    fig.add_trace(go.Scatter(x=[dates[i] for i in zero_crossings_up_trend], 
                            y=[usi_trend[i] for i in zero_crossings_up_trend], 
                            mode='markers', name='Bullish', marker=dict(color='green', size=10, symbol='triangle-up')), 
                            row=1, col=1)
    fig.add_trace(go.Scatter(x=[dates[i] for i in zero_crossings_down_trend], 
                            y=[usi_trend[i] for i in zero_crossings_down_trend], 
                            mode='markers', name='Bearish', marker=dict(color='red', size=10, symbol='triangle-down')), 
                            row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=usi_sma_trend, name='Trend (20-bar SMA)', 
                            line=dict(color='orange', dash='dot')), row=1, col=1)
    
    # Swing Trading (28 bars)
    usi_swing = calculate_usi(prices, period=28)
    zero_crossings_up_swing = np.where((usi_swing[:-1] < 0) & (usi_swing[1:] >= 0))[0] + 1
    zero_crossings_down_swing = np.where((usi_swing[:-1] > 0) & (usi_swing[1:] <= 0))[0] + 1
    usi_sma_swing = pd.Series(usi_swing).rolling(window=20, min_periods=1).mean().values
    
    fig.add_trace(go.Scatter(x=dates, y=usi_swing, name='USI (28 bars)', line=dict(color='blue')), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)
    fig.add_trace(go.Scatter(x=[dates[i] for i in zero_crossings_up_swing], 
                            y=[usi_swing[i] for i in zero_crossings_up_swing], 
                            mode='markers', name='Bullish', marker=dict(color='green', size=10, symbol='triangle-up')), 
                            row=2, col=1)
    fig.add_trace(go.Scatter(x=[dates[i] for i in zero_crossings_down_swing], 
                            y=[usi_swing[i] for i in zero_crossings_down_swing], 
                            mode='markers', name='Bearish', marker=dict(color='red', size=10, symbol='triangle-down')), 
                            row=2, col=1)
    fig.add_trace(go.Scatter(x=dates, y=usi_sma_swing, name='Trend (20-bar SMA)', 
                            line=dict(color='orange', dash='dot')), row=2, col=1)
    
    fig.update_layout(height=800, width=1000, 
                     title_text="USI: Trend vs. Swing Trading Comparison (SPY Data)",
                     template='plotly_white', showlegend=True)
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="USI Value", row=1, col=1)
    fig.update_yaxes(title_text="USI Value", row=2, col=1)
    
    fig.show()

if __name__ == "__main__":
    plot_side_by_side_comparison()