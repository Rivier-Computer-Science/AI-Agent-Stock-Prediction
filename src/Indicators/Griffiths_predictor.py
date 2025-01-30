import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
#import plotly.graph_objs as go
#from plotly.subplots import make_subplots
from src.Indicators.High_pass_filter_function import highpass_filter
from src.Indicators.SuperSmoother_filter_function import super_smoother

class DataFetcher:
    def __init__(self, symbol, start_date: datetime = None, end_date: datetime = None):
        self.symbol = symbol
        if start_date is None:
            self.start_date = datetime.today() - timedelta(days=60)
        else:
            self.start_date = start_date

        if end_date is None:
            self.end_date = datetime.today()
        else:
            self.end_date = end_date

    def get_stock_data(self):
        
        start_date_str = self.start_date.strftime('%Y-%m-%d')
        end_date_str = self.end_date.strftime('%Y-%m-%d')

        try:
        
            df = yf.download(self.symbol, start=start_date_str, end=end_date_str)
            if df.empty:
                print("No data fetched for symbol:", self.symbol)
                return None
            return df['Close']
        except Exception as e:
            print(f"Failed to fetch data: {e}")
            return None


def griffiths_predictor(close_prices, length=18, lower_bound=18, upper_bound=40, bars_fwd=2):
    mu = 1 / length
    hp = highpass_filter(close_prices, upper_bound)
    lp = super_smoother(hp, lower_bound)

    xx = np.zeros(length)  
    coef = np.zeros(length)  
    peak = 0.1
    predictions = np.zeros_like(close_prices)

    for t in range(length, len(lp)):
        if np.abs(lp[t]) > peak:
            peak = np.abs(lp[t])
        signal = lp[t] / peak if peak != 0 else 0

        xx[:-1] = xx[1:]
        xx[-1] = signal[-1]

        prediction = np.dot(xx, coef)
        predictions[t] = prediction

        error = signal - prediction
        coef += mu * error * xx

    future_signals = np.zeros(bars_fwd)
    for i in range(bars_fwd):
        future_signal = np.dot(xx, coef)
        future_signals[i] = future_signal
        xx[:-1] = xx[1:]  
        xx[-1] = future_signal 

    return predictions, future_signals


"""
# Usage example:
symbol = 'ES=F'
start_date = datetime(2023, 9, 1)
end_date = datetime(2024, 8, 31)
fetcher = DataFetcher(symbol=symbol, start_date=start_date, end_date=end_date)
price_data = fetcher.get_stock_data()

if price_data is not None:
    print(price_data.head())
    predicted_prices, future_prices = griffiths_predictor(price_data.values)
    print("Predicted data sample:", predicted_prices[:25])
    print("Length of price data:", len(price_data))
    print("Length of predicted signal:", len(predicted_prices))



    price_data.index = pd.to_datetime(price_data.index)

    # Debugging outputs
    print("Data length:", len(price_data))
    print("Date range in data:", price_data.index.min(), price_data.index.max())
    print("Data Index Type:", price_data.index.dtype)

    
    dates = pd.date_range(start='2023-09-01', periods=100, freq='D')
    original_prices = np.random.normal(5000, 200, size=100)  # Simulated price data
    predicted_prices = np.random.normal(0.3, 0.1, size=100)  # Simulated prediction data

    # Create DataFrame
    results_df = pd.DataFrame({
        'Date': dates,
        'Original Prices': original_prices,
        'Predicted Prices': predicted_prices
    })

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add trace for Original Prices on the primary y-axis
    fig.add_trace(
        go.Scatter(x=results_df['Date'], y=results_df['Original Prices'], name='Original Prices', mode='lines'),
        secondary_y=False  # Primary y-axis
    )

    # Add trace for Predicted Prices on the secondary y-axis
    fig.add_trace(
        go.Scatter(x=results_df['Date'], y=results_df['Predicted Prices'], name='Predicted Prices', mode='lines', line=dict(color='red')),
        secondary_y=True   # Secondary y-axis
    )

    # Update layout with titles and axis labels
    fig.update_layout(
        title='Comparison of Original and Predicted Stock Prices',
        xaxis_title='Date',
        yaxis_title='Original Prices ($)',
        yaxis2=dict(
            title='Predicted Prices',
            overlaying='y',
            side='right'
        ),
        template='plotly_dark'
    )

    # Show the figure
    fig.show()
    mse = np.mean(np.square(predicted_prices - price_data.values))
    print("Mean Squared Error:", mse)
else:
    print("No data available.")
"""