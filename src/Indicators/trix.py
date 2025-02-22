import pandas as pd

def calculate_trix(df: pd.DataFrame, length: int = 14, signal: int = 9) -> pd.DataFrame:
    """
    Calculates the TRIX (Triple-Smoothed Exponential Moving Average) indicator and
    its signal line.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'Close' prices.
        length (int): The EMA length for each of the triple smoothings.
        signal (int): The EMA period to compute the TRIX signal line.

    Returns:
        pd.DataFrame: The original DataFrame with two new columns:
                      - 'TRIX': TRIX values.
                      - 'TRIX_SIGNAL': Signal line of the TRIX.
    """
    # 1st smoothing
    ema1 = df['Close'].ewm(span=length, adjust=False).mean()
    # 2nd smoothing
    ema2 = ema1.ewm(span=length, adjust=False).mean()
    # 3rd smoothing
    ema3 = ema2.ewm(span=length, adjust=False).mean()

    # TRIX = 100 * (current_ema3 - previous_ema3) / previous_ema3
    df['TRIX'] = (ema3 - ema3.shift(1)) / ema3.shift(1) * 100

    # Signal line is often an EMA of the TRIX values
    df['TRIX_SIGNAL'] = df['TRIX'].ewm(span=signal, adjust=False).mean()

    return df
