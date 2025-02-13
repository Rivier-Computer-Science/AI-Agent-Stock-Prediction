from src.Agents.base_agent import BaseAgent
import numpy as np
import pandas as pd
from textwrap import dedent
from pydantic import PrivateAttr

class RiskMetricsKB(BaseAgent):
    _data: pd.DataFrame = PrivateAttr()
    _risk_metrics: dict = PrivateAttr()
    
    def __init__(self, data=None, **kwargs):
        super().__init__(
            role='Risk Metrics Analyst',
            goal=dedent("""
                Calculate and analyze portfolio risk metrics including VaR, 
                drawdowns, and volatility measures.
            """),
            backstory=dedent("""
                Expert in quantitative risk analysis with deep understanding 
                of market dynamics and portfolio risk assessment.
            """),
            **kwargs
        )
        self._data = data
        self._risk_metrics = {}
        
    def calculate_risk_metrics(self):
        """Calculate comprehensive risk metrics"""
        return crewai.Task(
            description=dedent("""
                Calculate key risk metrics for the portfolio including:
                - Value at Risk (VaR) at 95% and 99% confidence levels
                - Historical drawdowns
                - Rolling volatility
                - Asset correlations
            """),
            agent=self,
            expected_output="Dictionary containing calculated risk metrics"
        )
        
    def calculate_var(self, confidence_level=0.95):
        """Calculate Value at Risk"""
        try:
            returns = self._data['Close'].pct_change().dropna()
            var = np.percentile(returns, (1 - confidence_level) * 100)
            self.logger.info(f"Calculated VaR at {confidence_level*100}% confidence: {var:.4f}")
            return var
        except Exception as e:
            self.logger.error(f"Error calculating VaR: {str(e)}")
            return None
        
    def calculate_drawdown(self):
        """Calculate drawdown series"""
        try:
            prices = self._data['Close']
            rolling_max = prices.expanding().max()
            drawdown = (prices - rolling_max) / rolling_max
            self.logger.info(f"Max drawdown: {drawdown.min():.4f}")
            return drawdown
        except Exception as e:
            self.logger.error(f"Error calculating drawdown: {str(e)}")
            return None
        
    def calculate_correlation(self):
        """Calculate correlation matrix for asset returns"""
        try:
            returns = self._data['Close'].pct_change().dropna()
            correlation = returns.corr()
            self.logger.info("Correlation matrix calculated successfully")
            return correlation
        except Exception as e:
            self.logger.error(f"Error calculating correlation: {str(e)}")
            return None
        
    def calculate_volatility(self, window=30):
        """Calculate rolling volatility"""
        try:
            returns = self._data['Close'].pct_change().dropna()
            volatility = returns.rolling(window=window).std() * np.sqrt(252)
            self.logger.info(f"Calculated {window}-day rolling volatility")
            return volatility
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {str(e)}")
            return None
            
    def analyze_risk_metrics(self):
        """Analyze and validate risk metrics"""
        try:
            self._risk_metrics = {
                'var_95': self.calculate_var(0.95),
                'var_99': self.calculate_var(0.99),
                'drawdown': self.calculate_drawdown(),
                'volatility': self.calculate_volatility(),
                'correlation': self.calculate_correlation()
            }
            return self._risk_metrics
        except Exception as e:
            self.logger.error(f"Error analyzing risk metrics: {str(e)}")
            return None
