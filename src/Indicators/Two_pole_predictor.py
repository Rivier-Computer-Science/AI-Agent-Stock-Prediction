import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objs as go
from Filters.high_pass_2pole_filter import highpass_filter
from Filters.low_pass_2pole_filter import super_smoother

class DataFetcher:
    def __init__(self, symbol, start_date: datetime = None, end_date: datetime = None):
        self.symbol = symbol
        self.start_date = start_date if start_date else datetime.today() - timedelta(days=60)
        self.end_date = end_date if end_date else datetime.today()

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
            print("Failed to fetch data:", e)
            return None

def two_pole_predictor(price_series, q):


    hp = highpass_filter(price_series, 15)
    lp = super_smoother(hp, 30)
    
    c0 = 1
    c1 = 1.8 * q
    c2 = -q * q
    sum_coeffs = 1 - c1 - c2
    
    c0 /= sum_coeffs
    c1 /= sum_coeffs
    c2 /= sum_coeffs
    
    predicted = [0] * len(price_series)
    for i in range(2, len(price_series)):
        predicted[i] = c0 * lp[i] - c1 * lp[i - 1] - c2 * lp[i - 2]
    
    return predicted
"""
# Usage example:
symbol = 'ES=F'
start_date = datetime(2023, 9, 1)
end_date = datetime(2024, 8, 31)
fetcher = DataFetcher(symbol=symbol, start_date=start_date, end_date=end_date)
price_data = fetcher.get_stock_data()
if price_data is not None:
    q = 0.35
    predicted_prices = two_pole_predictor(price_data.squeeze().tolist(), q)

    original_prices_1d = np.array(price_data).flatten()
    predicted_prices_1d = np.array(predicted_prices).flatten()

    results_df = pd.DataFrame({
        'Date': price_data.index,
        'Original Prices': original_prices_1d,  # Ensured 1D
        'Predicted Prices': predicted_prices_1d  # Ensured 1D
    }, index=price_data.index)

    # Print the DataFrame
    print(results_df.head())

    # Plot the results
    trace1 = go.Scatter(x=results_df['Date'], y=results_df['Original Prices'], mode='lines', name='Original Prices')
    trace2 = go.Scatter(x=results_df['Date'], y=results_df['Predicted Prices'], mode='lines', name='Predicted Prices', line=dict(color='red'))

    fig = go.Figure(data=[trace1, trace2])
    fig.update_layout(
        title='Comparison of Original and Predicted Stock Prices',
        xaxis_title='Date',
        yaxis_title='Price',
        legend_title='Legend',
        template='plotly_dark'
    )
    fig.show()


"""