import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.Data_Retrieval.data_fetcher import DataFetcher
from src.Indicators.griffiths_predictor import GriffithsPredictor

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.Data_Retrieval.data_fetcher import DataFetcher
from src.Indicators.griffiths_predictor import GriffithsPredictor

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.Data_Retrieval.data_fetcher import DataFetcher
from src.Indicators.griffiths_predictor import GriffithsPredictor

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.Data_Retrieval.data_fetcher import DataFetcher
from src.Indicators.griffiths_predictor import GriffithsPredictor

def main():
    print("Fetching data for ticker AAPL")
    df = DataFetcher().get_stock_data(symbol="AAPL")
    close_prices = df["Close"]

    # Compute daily percent change and fill NaN with 0 so that the length is preserved.
    pct_change = close_prices.pct_change().fillna(0)
    date_index = pct_change.index

    # Instantiate the predictor with the close price series.
    gp = GriffithsPredictor(close_prices)

    # Get percent change predictions and future price forecast.
    pct_predictions, future_price = gp.predict_price()

    # Compute a historical predicted price series using positional indexing with .iloc.
    predicted_price = np.empty(len(close_prices))
    predicted_price[:gp.length] = np.nan  # No prediction for the first 'length' periods.
    for t in range(gp.length, len(close_prices)):
        # Use one-step forecast: predicted price at time t equals the previous close price times (1 + predicted percent change)
        predicted_price[t] = close_prices.iloc[t - 1] * (1 + pct_predictions[t])
    
    # Generate future dates for the future price forecast (assuming business days).
    last_date = date_index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                 periods=len(future_price),
                                 freq="B")

    # Create one figure with two subplots.
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 12), sharex=False)

    # Top subplot: Plot historical and predicted percent changes.
    axes[0].plot(date_index, pct_change.values, label="Historical Percent Change", color="blue")
    axes[0].plot(date_index, pct_predictions, label="Predicted Percent Change", color="orange", linestyle="--")
    axes[0].set_title("AAPL Daily Percent Change with Predictor")
    axes[0].set_ylabel("Percent Change")
    axes[0].legend()
    axes[0].grid(True)

    # Bottom subplot: Plot actual close prices, historical predicted price, and future price forecast.
    axes[1].plot(close_prices.index, close_prices.values, label="Actual Close Price", color="blue")
    axes[1].plot(close_prices.index, predicted_price, label="Historical Predicted Price", color="orange", linestyle="--")
    axes[1].plot(future_dates, future_price, label="Future Price Forecast", marker="o", linestyle="-", color="green")
    axes[1].set_title("AAPL Price Forecast with Predictor")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Price")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

