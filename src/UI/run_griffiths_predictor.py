import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.Data_Retrieval.data_fetcher import DataFetcher
from src.Indicators.griffiths_predictor import GriffithsPredictor


def main():
    ticker = "AAPL"
    df = DataFetcher().get_stock_data(symbol=ticker)
    close_prices = df["Close"]

    # Option 0: Use stationary (percent change) data.
    gp_pct_change = GriffithsPredictor(close_prices, make_stationary=True)
    pct_predictions, future_pct = gp_pct_change.predict_stationary()
    #pct_price_pred, future_price_from_pct = gp_pct_change.predict_price()

    # Option 1: Use original (price) data.
    gp_price = GriffithsPredictor(close_prices, make_stationary=False)
    price_predictions, future_price_direct = gp_price.predict_price()

    # Option 2: Use stationary (log difference) data
    gp_log_diff = GriffithsPredictor(close_prices, make_stationary=True, use_log_diff=True)
    log_predictions, future_log = gp_log_diff.predict_stationary()
    #log_price_pred, future_price_from_log = gp_log_diff.predict_price()

    
    # Generate future dates (assuming business days) for future predictions.
    # Assume future predictions (both pct and price) have the same length.
    num_future = len(future_pct)
    last_date = close_prices.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                 periods=num_future,
                                 freq="B")

    
    date_index = close_prices.index  

    # Plot Figure: Two subplots, top for percent change, bottom for price forecast.
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(14, 18), sharex=False)

    # Plot Price in Filter Space
    #axes[1].plot(date_index, gp_price.input_series, label="Close Price", color="blue")
    axes[0].plot(date_index, price_predictions, label="Predicted Price", color="orange", linestyle="--")
    axes[0].plot(future_dates, future_price_direct, label="Future Price Forecast", marker="o", linestyle="-", color="green")
    axes[0].set_title(f"{ticker} Price Forecast")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Price")
    axes[0].legend()
    axes[0].grid(True)

     # Plot Percent Change
    stationary_type = gp_pct_change.stationary_type
    axes[1].plot(date_index, gp_pct_change.input_series, label=f"{stationary_type}", color="blue")
    axes[1].plot(date_index, pct_predictions, label=f"Predicted {stationary_type}", color="orange", linestyle="--")
    axes[1].plot(future_dates, future_pct, label=f"Future Predicted {stationary_type}", marker="o", linestyle="-", color="green")
    axes[1].set_title(f"{ticker} Daily {stationary_type} (Stationary Predictor)")
    axes[1].set_ylabel(stationary_type)
    axes[1].legend()
    axes[1].grid(True)
 
    # Plot Log Difference
    stationary_type = gp_log_diff.stationary_type
    axes[2].plot(date_index, gp_log_diff.input_series, label=f"{stationary_type}", color="blue")
    axes[2].plot(date_index, log_predictions, label=f"Predicted {stationary_type}", color="orange", linestyle="--")
    axes[2].plot(future_dates, future_log, label=f"Future Predicted {stationary_type}", marker="o", linestyle="-", color="green")
    axes[2].set_title(f"{ticker} Daily {stationary_type} (Stationary Predictor)")
    axes[2].set_ylabel(stationary_type)
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

