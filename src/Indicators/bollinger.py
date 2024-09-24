import pandas as pd

class BollingerBands:
    def __init__(self, data, period=10, num_std=2):
        """
        Initializes the BollingerBands class.

        Args:
            data (pd.DataFrame): The stock data fetched from the DataFetcher.
            period (int): The period for calculating the rolling mean and standard deviation.
            num_std (int): The number of standard deviations to calculate the upper and lower bands.
        """
        self.data = data
        self.period = period
        self.num_std = num_std

    def calculate_bands(self):
        """
        Calculate Bollinger Bands for the provided stock data.

        Returns:
            dict: A dictionary containing the upper band, lower band, and moving average.
        """
        rolling_mean = self.data['Close'].rolling(window=self.period).mean()
        rolling_std = self.data['Close'].rolling(window=self.period).std()

        upper_band = rolling_mean + (rolling_std * self.num_std)
        lower_band = rolling_mean - (rolling_std * self.num_std)

        bands = {
            'Upper Band': upper_band,
            'Lower Band': lower_band,
            'Moving Average': rolling_mean
        }

        return bands
