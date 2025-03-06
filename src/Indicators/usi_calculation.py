import numpy as np
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Subtask 1: Develop SU/SD calculation using price data differences
def calculate_su_sd(prices):
    prices = np.array(prices)
    su = np.zeros(len(prices))
    sd = np.zeros(len(prices))
    
    for i in range(1, len(prices)):
        price_diff = prices[i] - prices[i-1]
        if price_diff > 0:
            su[i] = price_diff
        elif price_diff < 0:
            sd[i] = abs(price_diff)
            
    return su, sd

# Subtask 2: Apply UltimateSmoother filters to SU and SD values
def ultimate_smoother(price_series, period):
    price_series = np.array(price_series)
    us = np.zeros(len(price_series))
    
    a1 = np.exp(-1.414 * np.pi / period)
    b1 = 2 * a1 * np.cos(1.414 * np.pi / period)
    c2 = b1
    c3 = -a1 * a1
    c1 = (1 + c2 - c3) / 4
    
    us[:4] = price_series[:4]
    
    for i in range(4, len(price_series)):
        us[i] = (1 - c1) * price_series[i] + \
                (2 * c1 - c2) * price_series[i-1] - \
                (c1 + c3) * price_series[i-2] + \
                c2 * us[i-1] + \
                c3 * us[i-2]
    
    return us

def calculate_usi(su, sd, period, smoothing_period=4):
    su_avg = pd.Series(su).rolling(window=smoothing_period).mean().fillna(0).values
    sd_avg = pd.Series(sd).rolling(window=smoothing_period).mean().fillna(0).values
    
    usu = ultimate_smoother(su_avg, period)
    usd = ultimate_smoother(sd_avg, period)
    
    usi = np.zeros(len(su))
    for i in range(len(usi)):
        denominator = usu[i] + usd[i]
        if denominator != 0 and usu[i] > 0.01 and usd[i] > 0.01:
            usi[i] = (usu[i] - usd[i]) / denominator
    
    return usi

# Subtask 3: Validate USI outputs against traditional RSI results using pandas_ta
def calculate_traditional_rsi_pandasta(prices, period=14):
    df = pd.DataFrame({'close': prices})
    rsi = ta.rsi(df['close'], length=period)
    return rsi.values

# Example usage and validation with Plotly
if __name__ == "__main__":
    # Generate sample price data
    np.random.seed(42)
    prices = np.cumsum(np.random.normal(0, 1, 100)) + 100
    
    # Calculate SU and SD
    su, sd = calculate_su_sd(prices)
    
    # Calculate USI (using 28 bars as suggested in article)
    usi = calculate_usi(su, sd, period=28)
    
    # Calculate traditional RSI using pandas_ta for comparison
    rsi = calculate_traditional_rsi_pandasta(prices, period=14)
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Add USI trace
    fig.add_trace(
        go.Scatter(
            x=list(range(len(usi))),
            y=usi,
            name='USI (28 periods)',
            line=dict(color='blue', width=2),
            hovertemplate='USI: %{y:.4f}<br>Index: %{x}'
        )
    )
    
    # Add scaled RSI trace
    fig.add_trace(
        go.Scatter(
            x=list(range(len(rsi))),
            y=rsi/50 - 1,
            name='RSI/50 - 1 (14 periods)',
            line=dict(color='red', width=2),
            hovertemplate='RSI (scaled): %{y:.4f}<br>Index: %{x}'
        )
    )
    
    # Add zero line
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="black",
        annotation_text="Zero Line",
        annotation_position="bottom right"
    )
    
    # Update layout
    fig.update_layout(
        title='USI vs Traditional RSI ',
        xaxis_title='Time',
        yaxis_title='Indicator Value',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        template='plotly_white',
        hovermode='x unified',
        height=600,
        width=1000
    )
    
    # Add grid
    fig.update_yaxes(showgrid=True)
    fig.update_xaxes(showgrid=True)
    
    # Show the plot
    fig.show()