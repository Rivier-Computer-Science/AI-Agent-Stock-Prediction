# src/Indicators/adx_indicator.py
import pandas as pd
import numpy as np

class ADXIndicator:
    def __init__(self, period=14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Average Directional Index (ADX), along with +DI and -DI,
        using vectorized operations.

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
        # +DM: if the difference in high is greater than the absolute difference in low
        #      and is positive, use it; otherwise, 0.
        plus_dm = np.where((diff_high > diff_low.abs()) & (diff_high > 0), diff_high, 0)
        # -DM: if the absolute difference in low is greater than the difference in high
        #      and is positive, use it; otherwise, 0.
        minus_dm = np.where((diff_low.abs() > diff_high) & (diff_low.abs() > 0), diff_low.abs(), 0)

        df['+DM'] = plus_dm
        df['-DM'] = minus_dm

        # Smooth the TR, +DM, and -DM values using a rolling sum over the period
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

        # (Optional) Drop intermediate columns if you wish to keep only the final outputs:
        # df.drop(columns=['H-L', 'H-PC', 'L-PC', 'TR', '+DM', '-DM', 'TR_smooth',
        #                  '+DM_smooth', '-DM_smooth', '+DI', '-DI', 'DX'], inplace=True)

        return df
