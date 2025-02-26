from src.Agents.base_agent import BaseAgent
import crewai
from pydantic import PrivateAttr
from textwrap import dedent
import pandas as pd
import logging
from src.Indicators.risk_metrics_kb import RiskMetricsKB

class RiskAssessmentAgent(BaseAgent):
    """
    Risk Assessment Agent
    
    Analyzes portfolio risk metrics and provides comprehensive risk assessments
    using historical data, scenario analysis, and validation frameworks.
    """
    _portfolio_data: pd.DataFrame = PrivateAttr()
    _risk_metrics: RiskMetricsKB = PrivateAttr()
    _logger: logging.Logger = PrivateAttr()
    _asset_class_mapping: dict = PrivateAttr()
    
    def __init__(self, **kwargs):
        super().__init__(
            role='Risk Assessment Specialist',
            goal=dedent("""
                Analyze portfolio risk metrics and provide comprehensive risk assessments
                using historical data, scenario analysis, and validation frameworks.
            """),
            backstory=dedent("""
                Expert in financial risk assessment with extensive experience in
                portfolio analysis, risk metrics calculation, and scenario simulation.
                Specializes in stress testing and validation of risk models.
            """),
            **kwargs
        )
        self._portfolio_data = None
        self._risk_metrics = None
        self._logger = logging.getLogger("RiskAssessmentAgent")
        self._asset_class_mapping = {}
    
    def calculate_portfolio_risk(self, portfolio_data):
        """
        Calculate portfolio risk metrics task
        
        Args:
            portfolio_data: Portfolio data for risk calculation
            
        Returns:
            crewai.Task: Task for risk calculation
        """
        self._portfolio_data = portfolio_data
        self._risk_metrics = RiskMetricsKB(data=portfolio_data)
        
        return crewai.Task(
            description=dedent("""
                Calculate and analyze comprehensive portfolio risk metrics including:
                1. Value at Risk (VaR) at multiple confidence levels using different methods
                2. Historical drawdowns and recovery analysis
                3. Volatility measures and trends
                4. Asset correlations and diversification metrics
                5. Tail risk analysis including skewness and kurtosis
                6. Scenario-based stress testing
            """),
            agent=self,
            expected_output=dedent("""
                Dictionary containing:
                - VaR calculations at 95% and 99% confidence
                - Historical drawdown series and recovery analysis
                - Rolling volatility metrics
                - Correlation analysis
                - Tail risk metrics
                - Validation results
            """)
        )
    
    def analyze_asset_risks(self, portfolio_data=None, asset_class_mapping=None):
        """
        Analyze risk metrics by asset class
        
        Args:
            portfolio_data: Portfolio data (uses stored data if None)
            asset_class_mapping: Dictionary mapping symbols to asset classes
            
        Returns:
            dict: Asset class risk breakdown and analysis
        """
        if portfolio_data is not None:
            self._portfolio_data = portfolio_data
            
        if asset_class_mapping is not None:
            self._asset_class_mapping = asset_class_mapping
        
        if self._portfolio_data is None:
            self._logger.error("No portfolio data available for asset class analysis")
            return None
            
        if not self._asset_class_mapping:
            self._logger.warning("No asset class mapping provided")
            # Create default mapping based on symbol prefixes (simplified example)
            symbols = self._portfolio_data['symbol'].unique() if 'symbol' in self._portfolio_data.columns else []
            self._asset_class_mapping = {}
            
            for symbol in symbols:
                if symbol.startswith('B') or symbol.startswith('T'):
                    self._asset_class_mapping[symbol] = 'Bonds'
                elif symbol in ['GLD', 'SLV', 'IAU']:
                    self._asset_class_mapping[symbol] = 'Commodities'
                elif symbol in ['VNQ', 'IYR']:
                    self._asset_class_mapping[symbol] = 'Real Estate'
                else:
                    self._asset_class_mapping[symbol] = 'Equities'
        
        # Initialize risk metrics if needed
        if self._risk_metrics is None:
            self._risk_metrics = RiskMetricsKB(data=self._portfolio_data)
        
        # Calculate risk metrics by asset class
        asset_class_risks = self._risk_metrics.calculate_asset_class_risks(self._asset_class_mapping)
        
        if asset_class_risks is None:
            return None
        
        # Calculate portfolio-level metrics for comparison
        portfolio_var = self._risk_metrics.calculate_var(0.95)
        portfolio_vol = self._risk_metrics.calculate_volatility().iloc[-1] if self._risk_metrics.calculate_volatility() is not None else None
        
        # Calculate risk contribution of each asset class
        total_weight = sum(metrics['weight'] for _, metrics in asset_class_risks.items())
        
        for asset_class, metrics in asset_class_risks.items():
            # Normalize weight
            metrics['weight_normalized'] = metrics['weight'] / total_weight if total_weight > 0 else 0
            
            # Calculate risk contribution (simplified - assumes correlation = 1)
            if portfolio_vol is not None and portfolio_vol > 0:
                metrics['volatility_contribution'] = metrics['weight_normalized'] * metrics['volatility'] / portfolio_vol
            else:
                metrics['volatility_contribution'] = metrics['weight_normalized']
                
            # Calculate marginal VaR (simplified approach)
            metrics['var_contribution'] = metrics['weight_normalized'] * metrics['var_95'] / portfolio_var if portfolio_var else 0
        
        # Add portfolio-level metrics
        asset_class_risks['_portfolio'] = {
            'var_95': portfolio_var,
            'volatility': portfolio_vol,
            'weight': 1.0,
            'weight_normalized': 1.0,
            'volatility_contribution': 1.0,
            'var_contribution': 1.0
        }
        
        self._logger.info(f"Completed asset class risk analysis for {len(asset_class_risks)-1} asset classes")
        return asset_class_risks
    
    def validate_risk_metrics(self, risk_metrics=None, historical_data=None):
        """
        Validate calculated risk metrics
        
        Args:
            risk_metrics: Risk metrics to validate (uses stored metrics if None)
            historical_data: Historical data for validation (uses stored data if None)
            
        Returns:
            dict: Validation results and confidence metrics
        """
        if risk_metrics is None and self._risk_metrics is not None:
            # Use stored risk metrics results
            risk_metrics = self._risk_metrics.analyze_risk_metrics(validation=True)
            
        if risk_metrics is None:
            self._logger.error("No risk metrics available for validation")
            return None
            
        if historical_data is not None:
            self._portfolio_data = historical_data
            
        if self._portfolio_data is None:
            self._logger.error("No historical data available for validation")
            return None
        
        # Get validation results from risk metrics if available
        if 'validation' in risk_metrics:
            validation_results = risk_metrics['validation']
        else:
            # Initialize RiskMetricsKB if needed
            if self._risk_metrics is None:
                self._risk_metrics = RiskMetricsKB(data=self._portfolio_data)
                
            # Run validation
            validation_results = self._risk_metrics._validate_metrics()
        
        # Add backtesting validation
        backtesting_results = self._backtest_risk_metrics(risk_metrics)
        
        # Combine results
        combined_validation = {
            'metric_validation': validation_results,
            'backtesting': backtesting_results,
            'overall_confidence': 'High' if validation_results.get('validations_passed', False) and 
                                 backtesting_results.get('accuracy', 0) > 0.8 else 'Medium'
        }
        
        self._logger.info("Completed risk metrics validation")
        return combined_validation
    
    def _backtest_risk_metrics(self, risk_metrics):
        """
        Backtest risk metrics against historical data
        
        Args:
            risk_metrics: Risk metrics to backtest
            
        Returns:
            dict: Backtesting results
        """
        try:
            # Get historical data for backtesting
            if self._portfolio_data is None or len(self._portfolio_data) < 252:  # Need at least 1 year
                return {'accuracy': 0, 'error': "Insufficient historical data for backtesting"}
            
            # Backtest VaR
            var_accuracy = self._backtest_var(risk_metrics.get('var_95'))
            
            # Backtest volatility
            vol_accuracy = self._backtest_volatility(risk_metrics.get('volatility'))
            
            # Overall accuracy
            overall_accuracy = (var_accuracy + vol_accuracy) / 2
            
            return {
                'var_accuracy': var_accuracy,
                'volatility_accuracy': vol_accuracy,
                'accuracy': overall_accuracy,
                'sufficient_data': True
            }
            
        except Exception as e:
            self._logger.error(f"Error in backtesting: {str(e)}")
            return {'accuracy': 0, 'error': str(e)}
    
    def _backtest_var(self, var_95):
        """
        Backtest Value at Risk
        
        Args:
            var_95: 95% VaR value to test
            
        Returns:
            float: Accuracy score (0-1)
        """
        if var_95 is None or self._portfolio_data is None:
            return 0
            
        # Calculate daily returns
        returns = self._portfolio_data['Close'].pct_change().dropna()
        
        # Count exceptions (days where loss exceeded VaR)
        exceptions = (returns < var_95).sum()
        
        # Expected exceptions at 95% confidence: 5% of days
        expected_exceptions = len(returns) * 0.05
        
        # Calculate accuracy based on difference from expected exceptions
        exception_accuracy = 1 - min(1, abs(exceptions - expected_exceptions) / expected_exceptions)
        
        return exception_accuracy
    
    def _backtest_volatility(self, volatility_series):
        """
        Backtest volatility forecasts
        
        Args:
            volatility_series: Volatility series to test
            
        Returns:
            float: Accuracy score (0-1)
        """
        if volatility_series is None or len(volatility_series) < 30:
            return 0
            
        # Get realized volatility (simple calculation)
        returns = self._portfolio_data['Close'].pct_change().dropna()
        realized_vol = returns.rolling(30).std() * (252 ** 0.5)
        
        # Align forecast and realized
        common_idx = volatility_series.index.intersection(realized_vol.index)
        if len(common_idx) < 30:
            return 0
            
        forecast_vol = volatility_series.loc[common_idx]
        actual_vol = realized_vol.loc[common_idx]
        
        # Calculate mean absolute percentage error
        mape = (abs(forecast_vol - actual_vol) / actual_vol).mean()
        
        # Convert to accuracy score (0-1)
        accuracy = max(0, 1 - min(1, mape))
        
        return accuracy
    
    def run_stress_tests(self, scenarios=None):
        """
        Run multiple stress test scenarios
        
        Args:
            scenarios: List of scenario configurations
            
        Returns:
            dict: Stress test results for all scenarios
        """
        if self._portfolio_data is None:
            self._logger.error("No portfolio data available for stress testing")
            return None
            
        # Initialize risk metrics if needed
        if self._risk_metrics is None:
            self._risk_metrics = RiskMetricsKB(data=self._portfolio_data)
        
        # Default scenarios if none provided
        if scenarios is None:
            scenarios = [
                {"type": "historical", "params": {"period": "financial_crisis_2008"}},
                {"type": "historical", "params": {"period": "covid_crash_2020"}},
                {"type": "monte_carlo", "params": {"num_simulations": 1000, "distribution": "t"}},
                {"type": "custom", "params": {"shock_size": -0.15, "correlation_adjustment": 0.2}}
            ]
        
        # Run each scenario
        results = {}
        for i, scenario in enumerate(scenarios):
            scenario_type = scenario.get("type")
            params = scenario.get("params", {})
            
            scenario_result = self._risk_metrics.run_stress_test(scenario_type, **params)
            
            if scenario_result is not None:
                results[f"scenario_{i+1}"] = {
                    "configuration": scenario,
                    "results": scenario_result
                }
        
        # Add summary metrics
        worst_scenario = None
        worst_impact = 0
        
        for scenario_id, scenario_data in results.items():
            results = scenario_data.get("results", {})
            
            # Extract impact based on scenario type
            if results.get("stress_test_type") == "historical":
                impact = abs(results.get("max_drawdown", 0))
            elif results.get("stress_test_type") == "monte_carlo":
                impact = abs((results.get("percentiles", {}).get("p5", 0) / results.get("current_price", 1)) - 1)
            elif results.get("stress_test_type") == "custom":
                impact = abs(results.get("price_change", 0))
            else:
                impact = 0
                
            # Track worst scenario
            if impact > worst_impact:
                worst_impact = impact
                worst_scenario = scenario_id
                
            # Add impact to results
            scenario_data["impact"] = impact
            
        # Add summary
        summary = {
            "num_scenarios": len(results),
            "worst_scenario": worst_scenario,
            "worst_impact": worst_impact,
            "average_impact": sum(s.get("impact", 0) for s in results.values()) / len(results) if results else 0
        }
        
        self._logger.info(f"Completed {len(results)} stress test scenarios")
        
        return {
            "scenarios": results,
            "summary": summary
        }
    
    def get_risk_recommendations(self, risk_metrics=None, stress_test_results=None):
        """
        Generate risk management recommendations
        
        Args:
            risk_metrics: Risk metrics (uses stored metrics if None)
            stress_test_results: Stress test results
            
        Returns:
            dict: Risk management recommendations
        """
        if risk_metrics is None and self._risk_metrics is not None:
            risk_metrics = self._risk_metrics.analyze_risk_metrics()
            
        if risk_metrics is None:
            self._logger.error("No risk metrics available for recommendations")
            return None
            
        # Generate recommendations based on risk metrics and stress tests
        recommendations = []
        warning_level = "Low"  # Default
        
        # VaR-based recommendations
        var_95 = risk_metrics.get('var_95')
        if var_95 is not None:
            if var_95 < -0.03:  # More than 3% daily VaR
                recommendations.append(
                    "Consider reducing position sizes due to elevated Value at Risk."
                )
                warning_level = "High"
            elif var_95 < -0.02:  # 2-3% daily VaR
                recommendations.append(
                    "Monitor position sizes closely due to moderate Value at Risk."
                )
                warning_level = max(warning_level, "Medium")
        
        # Drawdown-based recommendations
        drawdown = risk_metrics.get('drawdown')
        if drawdown is not None:
            max_dd = drawdown.min()
            if max_dd < -0.2:  # More than 20% drawdown
                recommendations.append(
                    "Consider implementing stop-loss strategies due to significant historical drawdowns."
                )
                warning_level = "High"
            elif max_dd < -0.1:  # 10-20% drawdown
                recommendations.append(
                    "Review portfolio allocation to potentially reduce drawdown risk."
                )
                warning_level = max(warning_level, "Medium")
        
        # Volatility-based recommendations
        volatility = risk_metrics.get('volatility')
        if volatility is not None and len(volatility) > 0:
            current_vol = volatility.iloc[-1]
            if current_vol > 0.3:  # More than 30% annualized volatility
                recommendations.append(
                    "Consider diversification strategies to reduce current high volatility."
                )
                warning_level = "High"
            elif current_vol > 0.2:  # 20-30% annualized volatility
                recommendations.append(
                    "Monitor elevated volatility levels and consider hedging strategies."
                )
                warning_level = max(warning_level, "Medium")
        
        # Tail risk recommendations
        tail_risk = risk_metrics.get('tail_risk')
        if tail_risk is not None:
            skewness = tail_risk.get('skewness')
            kurtosis = tail_risk.get('kurtosis')
            
            if skewness is not None and skewness < -1.0:  # Significant negative skew
                recommendations.append(
                    "Consider tail risk hedging due to negatively skewed returns."
                )
                warning_level = max(warning_level, "Medium")
                
            if kurtosis is not None and kurtosis > 5.0:  # Fat tails
                recommendations.append(
                    "Portfolio exhibits fat-tailed return distribution. Consider options strategies to protect against extreme events."
                )
                warning_level = max(warning_level, "Medium")
        
        # Stress test recommendations
        if stress_test_results is not None:
            summary = stress_test_results.get('summary', {})
            worst_impact = summary.get('worst_impact', 0)
            
            if worst_impact > 0.3:  # More than 30% loss in worst scenario
                recommendations.append(
                    "Extreme stress test outcomes detected. Consider immediate portfolio rebalancing to reduce tail risk exposure."
                )
                warning_level = "High"
            elif worst_impact > 0.2:  # 20-30% loss in worst scenario
                recommendations.append(
                    "Significant stress test impacts. Review hedging strategies and consider adjustments."
                )
                warning_level = max(warning_level, "Medium")
        
        # Default recommendation if none generated
        if not recommendations:
            recommendations.append(
                "Portfolio risk metrics are within acceptable ranges. Continue regular monitoring."
            )
        
        return {
            "recommendations": recommendations,
            "warning_level": warning_level
        }
    
    def get_risk_metrics(self):
        """
        Get calculated risk metrics
        
        Returns:
            dict: Calculated risk metrics
        """
        if self._risk_metrics is None:
            self._logger.warning("Risk metrics not calculated yet")
            return None
            
        return self._risk_metrics.analyze_risk_metrics()
    
    def set_portfolio_data(self, portfolio_data):
        """
        Set portfolio data for analysis
        
        Args:
            portfolio_data: Portfolio data DataFrame
        """
        self._portfolio_data = portfolio_data
        self._risk_metrics = RiskMetricsKB(data=portfolio_data)
        
    def set_asset_class_mapping(self, mapping):
        """
        Set asset class mapping for risk breakdown
        
        Args:
            mapping: Dictionary mapping symbols to asset classes
        """
        self._asset_class_mapping = mapping
