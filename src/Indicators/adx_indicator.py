import pandas as pd
import numpy as np

class ADXIndicator:
    def __init__(self, period=14, smoothing_method="SMA"):
        """
        Initialize the ADXIndicator with a given period and smoothing method.
        
        Parameters:
            period (int): The time period for the ADX calculation.
            smoothing_method (str): The smoothing method to use ("SMA" or "EMA").
        """
        self.period = period
        self.smoothing_method = smoothing_method.upper()

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Average Directional Index (ADX), along with +DI and -DI.

        Parameters:
            data (pd.DataFrame): Historical data with 'High', 'Low', and 'Close' columns.

        Returns:
            pd.DataFrame: DataFrame with additional columns: +DM, -DM, TR, TR_smooth,
                          +DM_smooth, -DM_smooth, +DI, -DI, DX, and ADX.
        """
        df = data.copy()
        high = df['High']
        low = df['Low']
        close = df['Close']

        # Calculate True Range (TR)
        df['H-L'] = high - low
        df['H-PC'] = (high - close.shift(1)).abs()
        df['L-PC'] = (low - close.shift(1)).abs()
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)

        # Calculate directional movements using differences
        diff_high = high.diff()
        diff_low = low.diff()

        # Vectorized calculation for +DM and -DM:
        plus_dm = np.where((diff_high > diff_low.abs()) & (diff_high > 0), diff_high, 0)
        minus_dm = np.where((diff_low.abs() > diff_high) & (diff_low.abs() > 0), diff_low.abs(), 0)

        df['+DM'] = plus_dm
        df['-DM'] = minus_dm

        # Apply user-selected smoothing method
        if self.smoothing_method == "EMA":
            df['TR_smooth'] = df['TR'].ewm(span=self.period, min_periods=self.period, adjust=False).mean()
            df['+DM_smooth'] = df['+DM'].ewm(span=self.period, min_periods=self.period, adjust=False).mean()
            df['-DM_smooth'] = df['-DM'].ewm(span=self.period, min_periods=self.period, adjust=False).mean()
        else:  # Default to SMA
            df['TR_smooth'] = df['TR'].rolling(window=self.period, min_periods=self.period).sum()
            df['+DM_smooth'] = df['+DM'].rolling(window=self.period, min_periods=self.period).sum()
            df['-DM_smooth'] = df['-DM'].rolling(window=self.period, min_periods=self.period).sum()

        # Calculate the directional indicators +DI and -DI
        df['+DI'] = 100 * (df['+DM_smooth'] / df['TR_smooth'])
        df['-DI'] = 100 * (df['-DM_smooth'] / df['TR_smooth'])

        # Calculate the Directional Movement Index (DX)
        df['DX'] = 100 * (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI']))

        # Calculate the ADX as the rolling mean of DX over the period
        df['ADX'] = df['DX'].rolling(window=self.period, min_periods=self.period).mean()

        return df
