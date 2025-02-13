from crewai import Agent
from src.Agents.base_agent import BaseAgent
import numpy as np
import pandas as pd

class RiskAssessmentAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            role='Risk Assessment Specialist',
            goal='Analyze portfolio risk metrics and provide accurate risk assessments',
            backstory="""You are an expert in financial risk assessment with deep 
            knowledge of VaR calculations, drawdown analysis, and portfolio metrics."""
        )

    def calculate_portfolio_risk(self, portfolio_data):
        """Calculate comprehensive portfolio risk metrics"""
        try:
            returns = self._calculate_returns(portfolio_data)
            risk_metrics = {
                'var_95': self._calculate_var(returns, 0.95),
                'var_99': self._calculate_var(returns, 0.99),
                'drawdown': self._calculate_drawdown(portfolio_data),
                'volatility': self._calculate_volatility(returns),
                'sharpe_ratio': self._calculate_sharpe_ratio(returns)
            }
            return risk_metrics
        except Exception as e:
            self.verbose_print(f"Error calculating portfolio risk: {str(e)}")
            return None

    def analyze_asset_class_risks(self, portfolio_data):
        """Analyze risk metrics by asset class"""
        try:
            asset_classes = portfolio_data.groupby('asset_class')
            risk_by_class = {}
            
            for asset_class, data in asset_classes:
                returns = self._calculate_returns(data)
                risk_by_class[asset_class] = {
                    'var_95': self._calculate_var(returns, 0.95),
                    'volatility': self._calculate_volatility(returns)
                }
            return risk_by_class
        except Exception as e:
            self.verbose_print(f"Error analyzing asset class risks: {str(e)}")
            return None

    def validate_risk_metrics(self, risk_metrics, historical_data):
        try:
            validation_results = {
                'var_breaches': self._validate_var(risk_metrics['var_95'], historical_data),
                'volatility_stability': self._validate_volatility(risk_metrics['volatility'])
            }
            return validation_results
        except Exception as e:
            self.verbose_print(f"Error validating risk metrics: {str(e)}")
            return None

    def _calculate_returns(self, data):
        return data['Close'].pct_change().dropna()

    def _calculate_var(self, returns, confidence_level):
        return np.percentile(returns, (1 - confidence_level) * 100)

    def _calculate_drawdown(self, data):
        prices = data['Close']
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return drawdown.min()

    def _calculate_volatility(self, returns):
        return returns.std() * np.sqrt(252)

    def _calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        excess_returns = returns - risk_free_rate/252
        return np.sqrt(252) * excess_returns.mean() / returns.std()

    def _validate_var(self, var, historical_data):
        actual_returns = self._calculate_returns(historical_data)
        var_breaches = (actual_returns < var).mean()
        return var_breaches

    def _validate_volatility(self, volatility, window=30):
        """Validate volatility stability"""
        rolling_vol = volatility.rolling(window=window).std()
        return rolling_vol.mean()
