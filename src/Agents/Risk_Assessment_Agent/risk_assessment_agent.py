from src.Agents.base_agent import BaseAgent
import crewai as crewai
from pydantic import PrivateAttr
from textwrap import dedent
from src.Indicators.risk_metrics_kb import RiskMetricsKB

class RiskAssessmentAgent(BaseAgent):
    _portfolio_data: dict = PrivateAttr()
    _risk_metrics: RiskMetricsKB = PrivateAttr()
    
    def __init__(self, **kwargs):
        super().__init__(
            role='Risk Assessment Specialist',
            goal=dedent("""
                Analyze portfolio risk metrics and provide comprehensive risk assessments
                using historical data and scenario analysis.
            """),
            backstory=dedent("""
                Expert in financial risk assessment with extensive experience in
                portfolio analysis, risk metrics calculation, and scenario simulation.
            """),
            **kwargs
        )
    
    def calculate_portfolio_risk(self, portfolio_data):
        """Calculate portfolio risk metrics task"""
        self._portfolio_data = portfolio_data
        self._risk_metrics = RiskMetricsKB(data=portfolio_data)
        
        return crewai.Task(
            description=dedent("""
                Calculate and analyze comprehensive portfolio risk metrics including:
                1. Value at Risk (VaR) at multiple confidence levels
                2. Historical drawdowns and recovery analysis
                3. Volatility measures and trends
                4. Asset correlations and diversification metrics
            """),
            agent=self,
            expected_output=dedent("""
                Dictionary containing:
                - VaR calculations at 95% and 99% confidence
                - Historical drawdown series
                - Rolling volatility metrics
                - Correlation analysis
            """)
        )
    
    def analyze_asset_risks(self):
        """Analyze risk metrics by asset class"""
        return crewai.Task(
            description=dedent("""
                Break down risk metrics by asset class and analyze:
                1. Risk contribution of each asset class
                2. Correlation between asset classes
                3. Risk-adjusted performance metrics
            """),
            agent=self,
            expected_output="Asset class risk breakdown and analysis"
        )
    
    def validate_metrics(self):
        """Validate calculated risk metrics"""
        return crewai.Task(
            description=dedent("""
                Validate risk calculations through:
                1. Historical backtesting
                2. Scenario analysis comparison
                3. Metric stability assessment
            """),
            agent=self,
            expected_output="Validation results and confidence metrics"
        )

    def get_risk_metrics(self):
        """Get calculated risk metrics"""
        if hasattr(self, '_risk_metrics'):
            return self._risk_metrics.analyze_risk_metrics()
        return None
