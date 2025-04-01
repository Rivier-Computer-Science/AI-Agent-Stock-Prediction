import numpy as np
import pandas as pd

def ultimate_smoother(price, period):
    a1 = np.exp(-np.sqrt(2) * np.pi / period)
    b1 = 2 * a1 * np.cos(np.sqrt(2) * np.pi / period)
    c2 = b1
    c3 = -a1 ** 2
    c1 = 1 - c2 - c3

    filt = np.zeros_like(price)
    for i in range(2, len(price)):
        filt[i] = (c1 * price[i] + c2 * filt[i - 1] + c3 * filt[i - 2])
    return filt

def calculate_usi(df, length=14, window=4):
    prices = df['Close'].values
    su = np.maximum(prices - np.roll(prices, 1), 0)
    sd = np.maximum(np.roll(prices, 1) - prices, 0)

    su[0] = sd[0] = 0  # initial values

    usu = ultimate_smoother(pd.Series(su).rolling(window=window).mean().fillna(0).values, length)
    usd = ultimate_smoother(pd.Series(sd).rolling(window=window).mean().fillna(0).values, length)

    usi = np.where((usu + usd) != 0, (usu - usd) / (usu + usd), 0)

    return pd.Series(usi, index=df.index)