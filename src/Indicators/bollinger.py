import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import os

class BollingerBands:
    def __init__(self, data: pd.DataFrame, length: int=20, std: int=2):
        """
        Initializes the BollingerBands class.

        Args:
            data (pd.DataFrame): The stock data fetched from the DataFetcher.
            period (int): The period for calculating the rolling mean and standard deviation.
            num_std (int): The number of standard deviations to calculate the upper and lower bands.
        """
        self.data = data
        self.length = length
        self.std = std

    def calculate_bands(self) -> pd.DataFrame:
        """
        Calculate Bollinger Bands for the provided stock data.

        Returns:
            dict: A dictionary containing the upper band, lower band, and moving average.
        """
        df = self.data
        df.ta.bbands(close='Close', length=self.length, std=self.std, append=True)

        # Rename Columne
        suffix = f"{self.length}_{self.std:.1f}"
        rename_dict = {
            f"BBL_{suffix}": "Lower Band",
            f"BBM_{suffix}": "Moving Average",
            f"BBU_{suffix}": "Upper Band"
        }
        df.rename(columns=rename_dict, inplace=True)
        self.data = df.copy()
        
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print('calculate bolling bands:  \n',df)
        
        return df

    def manually_compute_buy_sell_hold_signals(self) -> pd.DataFrame:
        df = self.data
        df['Prev Close'] = df['Close'].shift(1)
        df['Prev Upper'] = df['Upper Band'].shift(1)
        df['Prev Lower'] = df['Lower Band'].shift(1)

        # Initialize Signal column
        df['Signal'] = 'HOLD'

        # SELL: Previous price above upper band, current price at or below
        sell_mask = (df['Prev Close'] > df['Prev Upper']) & (df['Close'] <= df['Upper Band'])

        # BUY: Previous price below lower band, current price at or above
        buy_mask = (df['Prev Close'] < df['Prev Lower']) & (df['Close'] >= df['Lower Band'])

        # Apply signals
        df.loc[sell_mask, 'Signal'] = 'SELL'
        df.loc[buy_mask, 'Signal'] = 'BUY'

        self.data = df.copy()
        return df
    
    def get_buy_sell_signals_drop_hold(self) -> pd.DataFrame:
        df = self.data.copy()
        df = df[df['Signal'].isin(['BUY', 'SELL'])]
        df = df.loc[:, ['Signal']]
        return df
    
    def plot_bollinger_band_data(self):
        os.environ["QT_QPA_PLATFORM"] = "xcb"   # For WSL
        df = self.data
        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(df['Close'], label='Close Price', color='black', linewidth=1.5)
        plt.plot(df['Upper Band'], label='Upper Band', color='red', linestyle='--')
        plt.plot(df['Moving Average'], label='Moving Average', color='blue')
        plt.plot(df['Lower Band'], label='Lower Band', color='green', linestyle='--')

        # Optional: Fill the band area
        plt.fill_between(df.index, df['Upper Band'], df['Lower Band'], color='gray', alpha=0.2)

        plt.title('Bollinger Bands with Price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()