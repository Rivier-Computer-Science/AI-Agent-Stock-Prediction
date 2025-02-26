from src.Agents.base_agent import BaseAgent
import numpy as np
import pandas as pd
from textwrap import dedent
from pydantic import PrivateAttr
import logging

class RiskMetricsKB(BaseAgent):
    """
    Risk Metrics Knowledge Base Agent
    
    Analyzes portfolio risk metrics including Value at Risk (VaR), drawdowns,
    volatility measures, and correlation analysis. Provides validation frameworks
    for risk metrics and supports scenario-based testing.
    """
    _data: pd.DataFrame = PrivateAttr()
    _risk_metrics: dict = PrivateAttr()
    _logger: logging.Logger = PrivateAttr()
    
    def __init__(self, data=None, **kwargs):
        super().__init__(
            role='Risk Metrics Analyst',
            goal=dedent("""
                Calculate and analyze comprehensive portfolio risk metrics including VaR, 
                drawdowns, volatility measures, and correlation analysis with validation frameworks.
            """),
            backstory=dedent("""
                Expert in quantitative risk analysis with deep understanding 
                of market dynamics, portfolio risk assessment, and statistical validation methods.
            """),
            **kwargs
        )
        self._data = data
        self._risk_metrics = {}
        self._logger = logging.getLogger("RiskMetricsKB")
        
    def calculate_risk_metrics(self):
        """Calculate comprehensive risk metrics task"""
        return self.create_task(
            description=dedent("""
                Calculate key risk metrics for the portfolio including:
                - Value at Risk (VaR) at 95% and 99% confidence levels
                - Conditional VaR (Expected Shortfall)
                - Historical drawdowns and recovery periods
                - Rolling volatility with different windows
                - Asset correlations and diversification metrics
                - Tail risk measures including skewness and kurtosis
            """),
            expected_output="Dictionary containing calculated risk metrics with validation metrics"
        )
        
    def calculate_var(self, confidence_level=0.95, method="historical", lookback_period=None):
        """
        Calculate Value at Risk using different methods
        
        Args:
            confidence_level: Confidence level for VaR calculation (0 to 1)
            method: Calculation method ("historical", "parametric", or "monte_carlo")
            lookback_period: Number of days to use for calculation (None for all data)
            
        Returns:
            Calculated VaR value as a decimal (e.g., 0.02 for 2% loss)
        """
        try:
            if self._data is None or len(self._data) == 0:
                self._logger.error("No data available for VaR calculation")
                return None
                
            returns = self._data['Close'].pct_change().dropna()
            
            # Limit to lookback period if specified
            if lookback_period and lookback_period < len(returns):
                returns = returns.iloc[-lookback_period:]
            
            if method == "historical":
                var = np.percentile(returns, (1 - confidence_level) * 100)
                
            elif method == "parametric":
                # Parametric method assumes normal distribution
                mean = returns.mean()
                std = returns.std()
                # Calculate Z-score for the confidence level
                from scipy import stats
                z_score = stats.norm.ppf(1 - confidence_level)
                var = mean + (z_score * std)
                
            elif method == "monte_carlo":
                # Monte Carlo simulation with 10,000 trials
                np.random.seed(42)  # For reproducibility
                mean = returns.mean()
                std = returns.std()
                simulations = 10000
                simulated_returns = np.random.normal(mean, std, simulations)
                var = np.percentile(simulated_returns, (1 - confidence_level) * 100)
                
            else:
                self._logger.error(f"Unknown VaR method: {method}")
                return None
                
            self._logger.info(f"Calculated VaR ({method}) at {confidence_level*100}% confidence: {var:.4f}")
            return var
            
        except Exception as e:
            self._logger.error(f"Error calculating VaR: {str(e)}")
            return None
    
    def calculate_cvar(self, confidence_level=0.95, method="historical", lookback_period=None):
        """
        Calculate Conditional Value at Risk (Expected Shortfall)
        
        Args:
            confidence_level: Confidence level for CVaR calculation (0 to 1)
            method: Calculation method ("historical" or "parametric")
            lookback_period: Number of days to use for calculation (None for all data)
            
        Returns:
            Calculated CVaR value as a decimal
        """
        try:
            if self._data is None or len(self._data) == 0:
                self._logger.error("No data available for CVaR calculation")
                return None
                
            returns = self._data['Close'].pct_change().dropna()
            
            # Limit to lookback period if specified
            if lookback_period and lookback_period < len(returns):
                returns = returns.iloc[-lookback_period:]
            
            var = self.calculate_var(confidence_level, method, lookback_period)
            
            if var is None:
                return None
                
            if method == "historical":
                # CVaR is the average of returns beyond VaR
                cvar = returns[returns <= var].mean()
                
            elif method == "parametric":
                # Parametric CVaR calculation
                from scipy import stats
                mean = returns.mean()
                std = returns.std()
                z_score = stats.norm.ppf(1 - confidence_level)
                cvar = mean - (std * stats.norm.pdf(z_score) / (1 - confidence_level))
                
            else:
                self._logger.error(f"Unsupported CVaR method: {method}")
                return None
                
            self._logger.info(f"Calculated CVaR at {confidence_level*100}% confidence: {cvar:.4f}")
            return cvar
            
        except Exception as e:
            self._logger.error(f"Error calculating CVaR: {str(e)}")
            return None
        
    def calculate_drawdown(self):
        """
        Calculate drawdown series
        
        Returns:
            pandas.Series: Drawdown values indexed by date
        """
        try:
            if self._data is None or len(self._data) == 0:
                self._logger.error("No data available for drawdown calculation")
                return None
                
            prices = self._data['Close']
            rolling_max = prices.expanding().max()
            drawdown = (prices - rolling_max) / rolling_max
            
            self._logger.info(f"Calculated drawdown series. Max drawdown: {drawdown.min():.4f}")
            return drawdown
            
        except Exception as e:
            self._logger.error(f"Error calculating drawdown: {str(e)}")
            return None
    
    def calculate_recovery_periods(self):
        """
        Calculate recovery periods from drawdowns
        
        Returns:
            list: List of drawdown periods with start/end dates and statistics
        """
        try:
            if self._data is None or len(self._data) == 0:
                self._logger.error("No data available for recovery period calculation")
                return None
                
            drawdown = self.calculate_drawdown()
            
            if drawdown is None:
                return None
                
            # Find drawdown periods
            drawdown_periods = []
            in_drawdown = False
            start_date = None
            max_drawdown = 0
            
            for date, dd_value in drawdown.items():
                # Start of drawdown period
                if not in_drawdown and dd_value < 0:
                    in_drawdown = True
                    start_date = date
                    max_drawdown = dd_value
                
                # Update max drawdown if in period
                elif in_drawdown and dd_value < max_drawdown:
                    max_drawdown = dd_value
                
                # End of drawdown period
                elif in_drawdown and dd_value >= 0:
                    in_drawdown = False
                    end_date = date
                    duration = (end_date - start_date).days
                    
                    drawdown_periods.append({
                        'start_date': start_date,
                        'end_date': end_date,
                        'duration_days': duration,
                        'max_drawdown': max_drawdown,
                        'recovery_days': None  # Will calculate in next step
                    })
            
            # Calculate recovery periods (time to return to previous peak)
            prices = self._data['Close']
            
            for period in drawdown_periods:
                if period['end_date'] >= prices.index[-1]:
                    # No recovery yet if drawdown continues to the end of the data
                    period['recovery_days'] = None
                    continue
                    
                start_price = prices.loc[period['start_date']]
                
                # Find first date after drawdown where price exceeds the start price
                recovery_slice = prices.loc[period['end_date']:] >= start_price
                recovery_dates = recovery_slice[recovery_slice].index
                
                if len(recovery_dates) > 0:
                    recovery_date = recovery_dates[0]
                    period['recovery_days'] = (recovery_date - period['end_date']).days
                else:
                    period['recovery_days'] = None
                    
            self._logger.info(f"Identified {len(drawdown_periods)} drawdown periods")
            return drawdown_periods
            
        except Exception as e:
            self._logger.error(f"Error calculating recovery periods: {str(e)}")
            return None
    
    def calculate_multi_asset_correlation(self, symbols=None):
        """
        Calculate correlation matrix for multiple assets
        
        Args:
            symbols: List of symbols to include (None for all)
            
        Returns:
            pd.DataFrame: Correlation matrix
        """
        try:
            if self._data is None or len(self._data) == 0:
                self._logger.error("No data available for correlation calculation")
                return None
            
            # Check if we have a multi-symbol dataframe
            if 'symbol' in self._data.columns:
                # Filter to specified symbols if provided
                if symbols is not None:
                    filtered_data = self._data[self._data['symbol'].isin(symbols)]
                else:
                    filtered_data = self._data.copy()
                
                # Pivot to get symbol returns
                pivot_data = filtered_data.pivot(index='date', columns='symbol', values='Close')
                returns = pivot_data.pct_change().dropna()
                
                # Calculate correlation matrix
                correlation = returns.corr()
                
                self._logger.info(f"Calculated correlation matrix for {len(correlation.columns)} symbols")
                return correlation
            
            # Single asset case - no correlation possible
            else:
                self._logger.warning("Correlation calculation requires multiple assets")
                return None
            
        except Exception as e:
            self._logger.error(f"Error calculating multi-asset correlation: {str(e)}")
            return None
    
    def calculate_single_asset_correlation(self, window=30):
        """
        Calculate rolling auto-correlation for a single asset
        
        Args:
            window: Window size for rolling auto-correlation
            
        Returns:
            pd.Series: Rolling auto-correlation series
        """
        try:
            if self._data is None or len(self._data) == 0:
                self._logger.error("No data available for auto-correlation calculation")
                return None
            
            returns = self._data['Close'].pct_change().dropna()
            
            # Calculate rolling auto-correlation (lag 1)
            rolling_autocorr = returns.rolling(window=window).apply(
                lambda x: x.iloc[:-1].corr(x.iloc[1:]) if len(x) > 1 else np.nan
            )
            
            self._logger.info(f"Calculated {window}-day rolling auto-correlation")
            return rolling_autocorr
            
        except Exception as e:
            self._logger.error(f"Error calculating auto-correlation: {str(e)}")
            return None
        
    def calculate_volatility(self, window=30, annualize=True):
        """
        Calculate rolling volatility
        
        Args:
            window: Window size for rolling volatility
            annualize: Whether to annualize the volatility (True/False)
            
        Returns:
            pd.Series: Rolling volatility series
        """
        try:
            if self._data is None or len(self._data) == 0:
                self._logger.error("No data available for volatility calculation")
                return None
                
            returns = self._data['Close'].pct_change().dropna()
            volatility = returns.rolling(window=window).std()
            
            if annualize:
                # Assuming daily data, annualize by multiplying by sqrt(252)
                volatility = volatility * np.sqrt(252)
                
            self._logger.info(f"Calculated {window}-day rolling volatility")
            return volatility
            
        except Exception as e:
            self._logger.error(f"Error calculating volatility: {str(e)}")
            return None
    
    def calculate_tail_risk(self):
        """
        Calculate tail risk measures including skewness and kurtosis
        
        Returns:
            dict: Dictionary containing tail risk metrics
        """
        try:
            if self._data is None or len(self._data) == 0:
                self._logger.error("No data available for tail risk calculation")
                return None
                
            returns = self._data['Close'].pct_change().dropna()
            
            # Calculate skewness and kurtosis
            skewness = returns.skew()
            kurtosis = returns.kurtosis()
            
            # Create tail risk score
            # Negative skewness increases risk, positive kurtosis indicates fat tails
            tail_risk_score = abs(min(0, skewness)) + max(0, kurtosis) / 3
            
            tail_risk = {
                'skewness': skewness,
                'kurtosis': kurtosis,
                'tail_risk_score': tail_risk_score  
            }
            
            self._logger.info(f"Calculated tail risk: skewness={skewness:.4f}, kurtosis={kurtosis:.4f}")
            return tail_risk
            
        except Exception as e:
            self._logger.error(f"Error calculating tail risk: {str(e)}")
            return None
    
    def calculate_asset_class_risks(self, asset_class_mapping=None):
        """
        Calculate risk metrics by asset class
        
        Args:
            asset_class_mapping: Dictionary mapping symbols to asset classes
            
        Returns:
            dict: Dictionary of risk metrics by asset class
        """
        try:
            if self._data is None or len(self._data) == 0:
                self._logger.error("No data available for asset class risk calculation")
                return None
            
            if 'symbol' not in self._data.columns:
                self._logger.warning("Asset class risk requires multi-symbol data")
                return None
            
            if asset_class_mapping is None:
                self._logger.warning("No asset class mapping provided")
                return None
            
            # Group data by asset class
            self._data['asset_class'] = self._data['symbol'].map(asset_class_mapping)
            
            # Calculate risk metrics for each asset class
            asset_class_risks = {}
            
            for asset_class in self._data['asset_class'].unique():
                class_data = self._data[self._data['asset_class'] == asset_class]
                
                # Calculate aggregated metrics for this asset class
                class_returns = class_data.groupby('date')['Close'].mean().pct_change().dropna()
                
                var_95 = np.percentile(class_returns, 5)
                volatility = class_returns.std() * np.sqrt(252)
                
                # Store metrics for this asset class
                asset_class_risks[asset_class] = {
                    'var_95': var_95,
                    'volatility': volatility,
                    'weight': len(class_data) / len(self._data)  # Approximate weight by count
                }
            
            self._logger.info(f"Calculated risk metrics for {len(asset_class_risks)} asset classes")
            return asset_class_risks
            
        except Exception as e:
            self._logger.error(f"Error calculating asset class risks: {str(e)}")
            return None
            
    def analyze_risk_metrics(self, validation=True):
        """
        Analyze and validate comprehensive risk metrics
        
        Args:
            validation: Whether to validate the calculated metrics
            
        Returns:
            dict: Dictionary containing risk metrics and validation results
        """
        try:
            if self._data is None or len(self._data) == 0:
                self._logger.error("No data available for risk metrics analysis")
                return None
                
            # Calculate comprehensive risk metrics
            self._risk_metrics = {
                'var_95': self.calculate_var(0.95, "historical"),
                'var_99': self.calculate_var(0.99, "historical"),
                'var_95_parametric': self.calculate_var(0.95, "parametric"),
                'cvar_95': self.calculate_cvar(0.95, "historical"),
                'cvar_99': self.calculate_cvar(0.99, "historical"),
                'drawdown': self.calculate_drawdown(),
                'recovery_periods': self.calculate_recovery_periods(),
                'volatility': self.calculate_volatility(30),
                'volatility_60d': self.calculate_volatility(60),
                'tail_risk': self.calculate_tail_risk(),
                'autocorrelation': self.calculate_single_asset_correlation(30)
            }
            
            # Validate metrics if requested
            if validation:
                validation_results = self._validate_metrics()
                self._risk_metrics['validation'] = validation_results
            
            return self._risk_metrics
            
        except Exception as e:
            self._logger.error(f"Error analyzing risk metrics: {str(e)}")
            return None
    
    def _validate_metrics(self):
        """
        Validate calculated risk metrics
        
        Returns:
            dict: Validation results
        """
        try:
            validation_results = {
                'issues': [],
                'warnings': [],
                'validations_passed': True
            }
            
            # Validate VaR
            var_95 = self._risk_metrics.get('var_95')
            var_95_parametric = self._risk_metrics.get('var_95_parametric')
            
            if var_95 is not None:
                # VaR should typically be negative (loss)
                if var_95 > 0:
                    validation_results['warnings'].append("VaR is positive, which is unusual")
                
                # Compare methods for consistency
                if var_95_parametric is not None:
                    var_diff = abs(var_95 - var_95_parametric)
                    if var_diff > 0.05:  # 5% difference threshold
                        validation_results['warnings'].append(
                            f"Large discrepancy between historical and parametric VaR: {var_diff:.4f}"
                        )
            
            # Validate CVaR
            cvar_95 = self._risk_metrics.get('cvar_95')
            if var_95 is not None and cvar_95 is not None:
                # CVaR should be more extreme than VaR
                if cvar_95 > var_95:
                    validation_results['issues'].append(
                        "CVaR is less extreme than VaR, which is inconsistent"
                    )
                    validation_results['validations_passed'] = False
            
            # Validate volatility
            volatility = self._risk_metrics.get('volatility')
            if volatility is not None:
                # Check for unrealistically high volatility
                if volatility.max() > 1.0:  # 100% annualized
                    validation_results['warnings'].append(
                        f"Extremely high volatility detected: {volatility.max():.2%}"
                    )
            
            # Validate tail risk
            tail_risk = self._risk_metrics.get('tail_risk')
            if tail_risk is not None:
                skewness = tail_risk.get('skewness')
                kurtosis = tail_risk.get('kurtosis')
                
                # Check for extreme values
                if abs(skewness) > 3:
                    validation_results['warnings'].append(
                        f"Extreme skewness detected: {skewness:.4f}"
                    )
                
                if kurtosis > 10:
                    validation_results['warnings'].append(
                        f"Extreme kurtosis detected: {kurtosis:.4f}"
                    )
            
            self._logger.info(
                f"Validation complete: {len(validation_results['issues'])} issues, "
                f"{len(validation_results['warnings'])} warnings"
            )
            
            return validation_results
            
        except Exception as e:
            self._logger.error(f"Error validating metrics: {str(e)}")
            return {
                'issues': [f"Validation error: {str(e)}"],
                'warnings': [],
                'validations_passed': False
            }
    
    def run_stress_test(self, scenario_type="historical", **kwargs):
        """
        Run stress test simulation on portfolio data
        
        Args:
            scenario_type: Type of stress test ("historical", "monte_carlo", "custom")
            **kwargs: Additional parameters for specific scenarios
                - For "historical": period (str) - e.g., "financial_crisis_2008"
                - For "monte_carlo": num_simulations (int), distribution (str)
                - For "custom": shock_size (float), correlation_adjustment (float)
                
        Returns:
            dict: Stress test results
        """
        try:
            if self._data is None or len(self._data) == 0:
                self._logger.error("No data available for stress testing")
                return None
                
            if scenario_type == "historical":
                period = kwargs.get("period", "financial_crisis_2008")
                return self._historical_stress_test(period)
                
            elif scenario_type == "monte_carlo":
                num_simulations = kwargs.get("num_simulations", 1000)
                distribution = kwargs.get("distribution", "normal")
                return self._monte_carlo_stress_test(num_simulations, distribution)
                
            elif scenario_type == "custom":
                shock_size = kwargs.get("shock_size", -0.15)
                correlation_adjustment = kwargs.get("correlation_adjustment", 0.2)
                return self._custom_stress_test(shock_size, correlation_adjustment)
                
            else:
                self._logger.error(f"Unknown scenario type: {scenario_type}")
                return None
                
        except Exception as e:
            self._logger.error(f"Error running stress test: {str(e)}")
            return None
    
    def _historical_stress_test(self, period):
        """
        Run historical stress test based on known market crashes
        
        Args:
            period: Historical period to simulate
            
        Returns:
            dict: Stress test results
        """
        # Historical crisis periods with approximate drawdowns
        crisis_scenarios = {
            "financial_crisis_2008": {"drawdown": -0.56, "duration_days": 517},
            "covid_crash_2020": {"drawdown": -0.35, "duration_days": 33},
            "dotcom_crash_2000": {"drawdown": -0.49, "duration_days": 929},
            "black_monday_1987": {"drawdown": -0.22, "duration_days": 3},
            "global_financial_crisis_2008": {"drawdown": -0.56, "duration_days": 517}
        }
        
        if period not in crisis_scenarios:
            self._logger.error(f"Unknown historical period: {period}")
            return None
        
        scenario = crisis_scenarios[period]
        
        # Get current portfolio value from latest close
        current_value = self._data['Close'].iloc[-1]
        
        # Simulate the impact of the historical scenario
        scenario_impact = current_value * scenario["drawdown"]
        scenario_value = current_value + scenario_impact
        
        # Calculate recovery based on historical pattern
        recovery_days = scenario["duration_days"]
        
        # Create projected price series
        import pandas as pd
        last_date = self._data.index[-1]
        scenario_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), 
            periods=recovery_days
        )
        
        # Simple linear path for drawdown (could be enhanced with actual patterns)
        drawdown_path = np.linspace(0, scenario["drawdown"], recovery_days)
        projected_prices = current_value * (1 + drawdown_path)
        
        # Calculate updated risk metrics under this scenario
        scenario_var = self.calculate_var() * (1 + scenario["drawdown"] / 2)
        
        results = {
            "stress_test_type": "historical",
            "period": period,
            "max_drawdown": scenario["drawdown"],
            "duration_days": scenario["duration_days"],
            "current_value": current_value,
            "scenario_value": scenario_value,
            "value_impact": scenario_impact,
            "scenario_var": scenario_var,
            "projected_prices": list(projected_prices),
            "projected_dates": [d.strftime("%Y-%m-%d") for d in scenario_dates]
        }
        
        self._logger.info(f"Completed historical stress test for {period}")
        return results
    
    def _monte_carlo_stress_test(self, num_simulations, distribution):
        """
        Run Monte Carlo stress test with specified distribution
        
        Args:
            num_simulations: Number of simulation paths
            distribution: Statistical distribution to use
            
        Returns:
            dict: Stress test results
        """
        np.random.seed(42)  # For reproducibility
        
        # Get historical parameters
        returns = self._data['Close'].pct_change().dropna()
        mean = returns.mean()
        std = returns.std()
        
        # Adjust for distribution type
        if distribution == "normal":
            # Standard normal distribution
            pass
        elif distribution == "t":
            # t-distribution with 5 degrees of freedom (fatter tails)
            import scipy.stats as stats
            mean = 0  # t-distribution is centered at 0
            std = stats.t.std(df=5)  # Adjust for t-distribution scale
        elif distribution == "skewed":
            # Skewed distribution (negative skew for more realistic market crashes)
            import scipy.stats as stats
            skew = -1.0  # Negative skew
            # Use skewed normal distribution
            mean = 0
            std = stats.skewnorm.std(a=skew)
        else:
            self._logger.error(f"Unknown distribution: {distribution}")
            return None
        
        # Simulation parameters
        horizon_days = 252  # 1 year forward simulation
        current_price = self._data['Close'].iloc[-1]
        
        # Run simulations
        all_paths = []
        
        for i in range(num_simulations):
            if distribution == "normal":
                daily_returns = np.random.normal(mean, std, horizon_days)
            elif distribution == "t":
                from scipy.stats import t
                daily_returns = t.rvs(df=5, loc=mean, scale=std, size=horizon_days)
            elif distribution == "skewed":
                from scipy.stats import skewnorm
                daily_returns = skewnorm.rvs(a=-1, loc=mean, scale=std, size=horizon_days)
            
            # Calculate cumulative path
            price_path = [current_price]
            for ret in daily_returns:
                price_path.append(price_path[-1] * (1 + ret))
            
            all_paths.append(price_path)
        
        # Extract key metrics from simulations
        all_paths_array = np.array(all_paths)
        
        # Calculate percentiles at final horizon
        final_values = all_paths_array[:, -1]
        percentiles = {
            "p1": np.percentile(final_values, 1),
            "p5": np.percentile(final_values, 5),
            "p10": np.percentile(final_values, 10),
            "p50": np.percentile(final_values, 50),
            "p90": np.percentile(final_values, 90),
            "p95": np.percentile(final_values, 95),
            "p99": np.percentile(final_values, 99)
        }
        
        # Calculate worst case path (1st percentile)
        worst_path_idx = np.argmin(final_values)
        worst_path = all_paths[worst_path_idx]
        
        # Calculate median path
        median_path_idx = np.argmin(np.abs(final_values - percentiles["p50"]))
        median_path = all_paths[median_path_idx]
        
        # Generate dates for the projection
        last_date = self._data.index[-1]
        projection_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=horizon_days + 1
        )
        
        results = {
            "stress_test_type": "monte_carlo",
            "distribution": distribution,
            "num_simulations": num_simulations,
            "horizon_days": horizon_days,
            "current_price": current_price,
            "percentiles": percentiles,
            "worst_case": {
                "path": worst_path,
                "final_value": worst_path[-1],
                "return": (worst_path[-1] / current_price) - 1
            },
            "median_case": {
                "path": median_path,
                "final_value": median_path[-1],
                "return": (median_path[-1] / current_price) - 1
            },
            "projection_dates": [d.strftime("%Y-%m-%d") for d in projection_dates]
        }
        
        self._logger.info(f"Completed Monte Carlo stress test with {num_simulations} simulations")
        return results
    
    def _custom_stress_test(self, shock_size, correlation_adjustment):
        """
        Run custom stress test with specified parameters
        
        Args:
            shock_size: Size of price shock (negative for drops)
            correlation_adjustment: Correlation adjustment during stress
            
        Returns:
            dict: Stress test results
        """
        # Get current portfolio characteristics
        current_price = self._data['Close'].iloc[-1]
        current_vol = self.calculate_volatility().iloc[-1]
        
        # Calculate shocked values
        shocked_price = current_price * (1 + shock_size)
        
        # Assuming volatility increases during stress
        shocked_vol = current_vol * (1 + abs(shock_size))
        
        # Recalculate VaR with higher volatility to simulate stress
        current_var = self.calculate_var(0.95)
        shocked_var = current_var * (1 + abs(shock_size) * 1.5)  # VaR deteriorates more than linearly
        
        # Create recovery path (simplified linear recovery)
        recovery_days = int(abs(shock_size) * 100)  # Heuristic: 15% drop = 15 days recovery
        recovery_path = np.linspace(shocked_price, current_price, recovery_days)
        
        # Generate dates for the projection
        last_date = self._data.index[-1]
        projection_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=recovery_days
        )
        
        results = {
            "stress_test_type": "custom",
            "shock_size": shock_size,
            "correlation_adjustment": correlation_adjustment,
            "current_price": current_price,
            "shocked_price": shocked_price,
            "price_change": shock_size,
            "current_volatility": current_vol,
            "shocked_volatility": shocked_vol,
            "volatility_change": (shocked_vol / current_vol) - 1,
            "current_var": current_var,
            "shocked_var": shocked_var,
            "var_change": (shocked_var / current_var) - 1,
            "recovery_days": recovery_days,
            "recovery_path": list(recovery_path),
            "projection_dates": [d.strftime("%Y-%m-%d") for d in projection_dates]
        }
        
        self._logger.info(f"Completed custom stress test with shock size {shock_size}")
        return results
