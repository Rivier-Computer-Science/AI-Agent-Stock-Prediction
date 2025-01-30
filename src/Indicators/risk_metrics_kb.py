import numpy as np
import pandas as pd

class RiskMetrics:
    def __init__(self, data):
        self.data = data
        
    def calculate_var(self, confidence_level=0.95):
        """Calculate Value at Risk"""
        returns = self.data['Close'].pct_change().dropna()
        var = np.percentile(returns, (1 - confidence_level) * 100)
        return var
        
    def calculate_drawdown(self):
        """Calculate drawdown series"""
        prices = self.data['Close']
        rolling_max = prices.expanding().max()
        drawdown = (prices - rolling_max) / rolling_max
        return drawdown
        
    def calculate_correlation(self):
        """Calculate correlation matrix for asset returns"""
        returns = self.data['Close'].pct_change().dropna()
        correlation = returns.corr()
        return correlation
        
    def calculate_volatility(self, window=30):
        """Calculate rolling volatility"""
        returns = self.data['Close'].pct_change().dropna()
        volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
        return volatility
