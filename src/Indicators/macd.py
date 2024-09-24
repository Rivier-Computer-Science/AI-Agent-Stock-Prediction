import pandas as pd

class MACDIndicator:
    def __init__(self, period_short=12, period_long=26, signal_period=9):
        """
        Initializes the MACDIndicator class.

        Args:
            period_short (int): The short period for calculating the EMA (default is 12).
            period_long (int): The long period for calculating the EMA (default is 26).
            signal_period (int): The signal line period (default is 9).
        """
        self.period_short = period_short
        self.period_long = period_long
        self.signal_period = signal_period

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MACD using pandas.

        Args:
            data (pd.DataFrame): Stock data containing at least a 'Close' column.

        Returns:
            pd.DataFrame: DataFrame with MACD, Signal Line, and MACD Histogram.
        """
        # Ensure 'Close' column is available in the data
        if 'Close' not in data.columns:
            raise ValueError("Data must contain 'Close' prices.")

        # Calculate short-term EMA (12-period EMA)
        short_ema = data['Close'].ewm(span=self.period_short, adjust=False).mean()

        # Calculate long-term EMA (26-period EMA)
        long_ema = data['Close'].ewm(span=self.period_long, adjust=False).mean()

        # MACD line is the difference between short-term EMA and long-term EMA
        macd_line = short_ema - long_ema

        # Signal line is the 9-period EMA of the MACD line
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()

        # MACD histogram is the difference between MACD line and signal line
        macd_histogram = macd_line - signal_line

        # Combine everything into a DataFrame
        macd_df = pd.DataFrame({
            'MACD Line': macd_line,
            'Signal Line': signal_line,
            'MACD Histogram': macd_histogram
        })

        return macd_df
