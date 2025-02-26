import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy
if not hasattr(numpy, 'NaN'):
    numpy.NaN = numpy.nan

import streamlit as st
import pandas as pd
from crewai import Crew
from src.Data_Retrieval.data_fetcher import DataFetcher
from src.Indicators.sma import SMAIndicator
from src.Indicators.risk_metrics_kb import RiskMetricsKB
from src.Agents.Risk_Assessment_Agent.risk_assessment_agent import RiskAssessmentAgent
from src.Agents.Scenario_Agents.scenario_simulation_agent import ScenarioSimulationAgent
import pandas_ta as ta
import plotly.graph_objects as go
import plotly.express as px
import time

class EnhancedRiskDashboard:
    def __init__(self):
        self.risk_agent = RiskAssessmentAgent()
        self.scenario_agent = ScenarioSimulationAgent()
        self.data_fetcher = DataFetcher()
        
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state for persistent storage"""
        if 'favorite_symbols' not in st.session_state:
            st.session_state.favorite_symbols = []
            
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = {}
            
        if 'current_data' not in st.session_state:
            st.session_state.current_data = None
        
    def setup_crew(self, portfolio_data=None):
        return Crew(
            agents=[self.risk_agent, self.scenario_agent],
            tasks=[
                self.risk_agent.calculate_portfolio_risk(portfolio_data),
                self.scenario_agent.run_simulation()
            ]
        )
        
    def display_risk_metrics(self, data):
        """Display enhanced risk metrics with tabbed interface"""
        try:
   
            st.session_state.current_data = data
            symbol = data.get('symbol', 'Unknown')
            
            with st.spinner("Calculating risk metrics..."):
                self.risk_agent.set_portfolio_data(data)
                risk_metrics = self.risk_agent.get_risk_metrics()
                
                st.session_state.analysis_results[symbol] = risk_metrics
                
            if not risk_metrics:
                st.error("Failed to calculate risk metrics. Please check the data.")
                return
                
            tab1, tab2, tab3, tab4 = st.tabs([
                "VaR Analysis", 
                "Drawdown", 
                "Volatility", 
                "Tail Risk"
            ])
            
            with tab1:
                self._display_var_analysis(data, risk_metrics)
                
            with tab2:
                self._display_drawdown_analysis(data, risk_metrics)
                
            with tab3:
                self._display_volatility_analysis(data, risk_metrics)
                
            with tab4:
                self._display_tail_risk_analysis(data, risk_metrics)
                
            # Summary metrics table
            st.subheader('Risk Metrics Summary')
            self._display_metrics_summary(risk_metrics)
                
        except Exception as e:
            st.error(f"Error displaying risk metrics: {str(e)}")
            
    def _display_var_analysis(self, data, risk_metrics):
        """Display Value at Risk analysis"""
        st.subheader('Value at Risk (VaR) Analysis')
        
        # Create VaR visualization
        returns = data['Close'].pct_change().dropna()
        
        # Get VaR values from different methods
        var_hist = risk_metrics.get('var_95')
        var_param = risk_metrics.get('var_95_parametric')
        cvar = risk_metrics.get('cvar_95')
        
        # Create histogram with VaR lines
        fig = go.Figure()
        
        # Returns distribution
        fig.add_trace(go.Histogram(
            x=returns,
            name='Returns Distribution',
            opacity=0.7,
            nbinsx=30
        ))
        
        # Historical VaR line
        if var_hist is not None:
            fig.add_vline(
                x=var_hist,
                line_width=2,
                line_dash="dash",
                line_color="red",
                annotation_text=f"VaR (95%): {var_hist:.2%}",
                annotation_position="top right"
            )
        
        # Parametric VaR line
        if var_param is not None:
            fig.add_vline(
                x=var_param,
                line_width=2,
                line_dash="dot",
                line_color="orange",
                annotation_text=f"Parametric VaR: {var_param:.2%}",
                annotation_position="bottom right"
            )
        
        # CVaR line
        if cvar is not None:
            fig.add_vline(
                x=cvar,
                line_width=2,
                line_dash="dash",
                line_color="darkred",
                annotation_text=f"CVaR (95%): {cvar:.2%}",
                annotation_position="bottom left"
            )
        
        fig.update_layout(
            title='Value at Risk Analysis',
            xaxis_title='Returns',
            yaxis_title='Frequency',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Explanation
        with st.expander("What is Value at Risk (VaR)?"):
            st.write("""
            **Value at Risk (VaR)** represents the maximum expected loss over a specific time horizon at a 
            given confidence level. For example, a 95% daily VaR of -2% means there's a 95% chance 
            that your loss won't exceed 2% over a one-day period.
            
            **Conditional VaR (CVaR)**, also known as Expected Shortfall, measures the average loss 
            in the worst cases beyond the VaR threshold, providing insight into tail risk.
            """)
            
    def _display_drawdown_analysis(self, data, risk_metrics):
        """Display drawdown analysis"""
        st.subheader('Historical Drawdown Analysis')
        
        drawdown = risk_metrics.get('drawdown')
        recovery_periods = risk_metrics.get('recovery_periods')
        
        if drawdown is None:
            st.warning("Drawdown data not available")
            return
            
        # Create drawdown visualization
        fig = go.Figure()
        
        # Drawdown area chart
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            fill='tozeroy',
            name='Drawdown',
            line=dict(color='rgba(220, 53, 69, 0.8)')
        ))
        
        fig.update_layout(
            title='Historical Drawdown Analysis',
            xaxis_title='Date',
            yaxis_title='Drawdown',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Drawdown statistics
        if recovery_periods and len(recovery_periods) > 0:
            st.subheader("Major Drawdown Events")
            
            # Convert to DataFrame for display
            df_periods = pd.DataFrame(recovery_periods)
            
            # Format for display
            if 'max_drawdown' in df_periods.columns:
                df_periods['max_drawdown'] = df_periods['max_drawdown'].apply(lambda x: f"{x:.2%}")
                
            if 'start_date' in df_periods.columns and 'end_date' in df_periods.columns:
                df_periods['period'] = df_periods.apply(
                    lambda row: f"{row['start_date'].strftime('%Y-%m-%d')} to {row['end_date'].strftime('%Y-%m-%d')}",
                    axis=1
                )
                
                # Keep only relevant columns
                df_display = df_periods[['period', 'max_drawdown', 'duration_days', 'recovery_days']].copy()
                df_display.columns = ['Period', 'Max Drawdown', 'Duration (Days)', 'Recovery (Days)']
                
                st.table(df_display)
        
        # Explanation
        with st.expander("What is Drawdown?"):
            st.write("""
            **Drawdown** measures the peak-to-trough decline in an investment's value, typically expressed 
            as a percentage from the peak. It helps assess the historical risk of an investment and 
            understand the magnitude of past declines.
            
            **Recovery period** is the time it takes for an investment to return to its previous peak 
            after a drawdown, providing insight into the resilience of the investment.
            """)
            
    def _display_volatility_analysis(self, data, risk_metrics):
        """Display volatility analysis"""
        st.subheader('Volatility Analysis')
        
        volatility = risk_metrics.get('volatility')
        vol_60d = risk_metrics.get('volatility_60d')
        
        if volatility is None:
            st.warning("Volatility data not available")
            return
            
        # Create volatility visualization
        fig = go.Figure()
        
        # Main volatility
        fig.add_trace(go.Scatter(
            x=volatility.index,
            y=volatility.values,
            name='30-Day Volatility',
            line=dict(color='rgba(66, 135, 245, 1)')
        ))
        
        # Add longer-term volatility if available
        if vol_60d is not None:
            fig.add_trace(go.Scatter(
                x=vol_60d.index,
                y=vol_60d.values,
                name='60-Day Volatility',
                line=dict(color='rgba(79, 61, 207, 0.7)')
            ))
        
        fig.update_layout(
            title='Annualized Rolling Volatility Analysis',
            xaxis_title='Date',
            yaxis_title='Annualized Volatility',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Explanation
        with st.expander("What is Volatility?"):
            st.write("""
            **Volatility** measures how much an asset's price fluctuates over time. Higher volatility 
            indicates larger price swings and potentially higher risk. It's typically calculated as 
            the standard deviation of returns and annualized to represent the expected yearly fluctuation.
            """)
            
    def _display_tail_risk_analysis(self, data, risk_metrics):
        """Display tail risk analysis"""
        st.subheader('Tail Risk Analysis')
        
        tail_risk = risk_metrics.get('tail_risk')
        
        if tail_risk is None:
            st.warning("Tail risk data not available")
            return
            
        # Get tail risk metrics
        skewness = tail_risk.get('skewness')
        kurtosis = tail_risk.get('kurtosis')
        tail_risk_score = tail_risk.get('tail_risk_score')
        
        # Create returns distribution visualization
        returns = data['Close'].pct_change().dropna()
        
        fig = go.Figure()
        
        # Returns histogram
        fig.add_trace(go.Histogram(
            x=returns,
            name='Returns Distribution',
            opacity=0.7,
            nbinsx=30,
            histnorm='probability'
        ))
        
        # Add normal distribution for comparison
        from scipy import stats
        
        mean = returns.mean()
        std = returns.std()
        x = numpy.linspace(mean - 4*std, mean + 4*std, 100)
        y = stats.norm.pdf(x, mean, std)
        
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            name='Normal Distribution',
            line=dict(color='rgba(54, 162, 235, 1)')
        ))
        
        fig.add_annotation(
            x=0.05,
            y=0.95,
            xref="paper",
            yref="paper",
            text=f"Skewness: {skewness:.4f}",
            showarrow=False,
            align="left"
        )
        
        fig.add_annotation(
            x=0.05,
            y=0.9,
            xref="paper",
            yref="paper",
            text=f"Kurtosis: {kurtosis:.4f}",
            showarrow=False,
            align="left"
        )
        
        fig.update_layout(
            title='Tail Risk Analysis - Returns Distribution',
            xaxis_title='Returns',
            yaxis_title='Probability',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Explanation
        with st.expander("What is Tail Risk?"):
            st.write("""
            **Tail risk** refers to the risk of extreme negative returns that fall in the "tails" of 
            the return distribution. It's measured through:
            
            - **Skewness**: Measures the asymmetry of returns. Negative skewness indicates 
              a distribution with a longer left tail, meaning more frequent small gains but occasional 
              large losses.
              
            - **Kurtosis**: Measures the "fatness" of the tails. Higher kurtosis indicates 
              more frequent extreme values than a normal distribution would predict.
            """)
            
    def _display_metrics_summary(self, risk_metrics):
        """Display risk metrics summary table"""
        
        # Prepare summary data
        var_95 = risk_metrics.get('var_95')
        cvar_95 = risk_metrics.get('cvar_95')
        drawdown = risk_metrics.get('drawdown')
        volatility = risk_metrics.get('volatility')
        tail_risk = risk_metrics.get('tail_risk', {})
        
        max_dd = drawdown.min() if drawdown is not None else None
        current_vol = volatility.iloc[-1] if volatility is not None and len(volatility) > 0 else None
        
        summary_data = {
            'Metric': [
                'VaR (95%)',
                'CVaR (95%)',
                'Max Drawdown',
                'Current Volatility',
                'Skewness',
                'Kurtosis',
                'Tail Risk Score'
            ],
            'Value': [
                f"{var_95:.2%}" if var_95 is not None else "N/A",
                f"{cvar_95:.2%}" if cvar_95 is not None else "N/A",
                f"{max_dd:.2%}" if max_dd is not None else "N/A",
                f"{current_vol:.2%}" if current_vol is not None else "N/A",
                f"{tail_risk.get('skewness', 'N/A'):.4f}" if 'skewness' in tail_risk else "N/A",
                f"{tail_risk.get('kurtosis', 'N/A'):.4f}" if 'kurtosis' in tail_risk else "N/A",
                f"{tail_risk.get('tail_risk_score', 'N/A'):.4f}" if 'tail_risk_score' in tail_risk else "N/A"
            ]
        }
        
        # Display as table
        st.table(pd.DataFrame(summary_data))
    
    def display_scenario_analysis(self, data):
        """Display scenario analysis using CrewAI agents"""
        st.subheader("Scenario Analysis")
        
        try:
            # Configure risk agent
            self.risk_agent.set_portfolio_data(data)
            
            # Scenario selection
            scenario_type = st.selectbox(
                "Select Scenario Type",
                ["Historical Crisis", "Market Shock", "Monte Carlo Simulation", "Custom"]
            )
            
            # Parameter inputs based on scenario type
            if scenario_type == "Historical Crisis":
                crisis = st.selectbox(
                    "Select Historical Crisis",
                    ["Financial Crisis (2008)", "COVID-19 Crash (2020)", "Dot-com Crash (2000)", "Black Monday (1987)"]
                )
                
                # Map to scenario params
                crisis_mapping = {
                    "Financial Crisis (2008)": "financial_crisis_2008",
                    "COVID-19 Crash (2020)": "covid_crash_2020",
                    "Dot-com Crash (2000)": "dotcom_crash_2000",
                    "Black Monday (1987)": "black_monday_1987"
                }
                
                params = {"period": crisis_mapping.get(crisis, "financial_crisis_2008")}
                scenario_config = {"type": "historical", "params": params}
                
            elif scenario_type == "Market Shock":
                shock_size = st.slider("Shock Size (%)", min_value=-50, max_value=0, value=-15, step=5)
                
                params = {"shock_size": shock_size/100}
                scenario_config = {"type": "custom", "params": params}
                
            elif scenario_type == "Monte Carlo Simulation":
                distribution = st.selectbox("Distribution Type", ["normal", "t", "skewed"])
                num_simulations = st.slider("Number of Simulations", min_value=100, max_value=5000, value=1000, step=100)
                
                params = {"distribution": distribution, "num_simulations": num_simulations}
                scenario_config = {"type": "monte_carlo", "params": params}
                
            else:  # Custom
                shock_size = st.slider("Price Shock (%)", min_value=-50, max_value=50, value=-15, step=5)
                vol_change = st.slider("Volatility Change (%)", min_value=-50, max_value=200, value=50, step=10)
                
                params = {
                    "shock_size": shock_size/100,
                    "vol_change": vol_change/100
                }
                scenario_config = {"type": "custom", "params": params}
                
            # Run scenario button
            if st.button("Run Scenario Analysis"):
                with st.spinner(f"Running {scenario_type} scenario analysis..."):
                    # Add progress bar
                    progress_bar = st.progress(0)
                    for i in range(100):
                        # Simulated progress updates
                        time.sleep(0.02)
                        progress_bar.progress(i + 1)
                
                    scenarios = [scenario_config]
                    stress_results = self.risk_agent.run_stress_tests(scenarios)
                    
                    if stress_results:
                        summary = stress_results.get('summary', {})
                        st.subheader("Scenario Analysis Results")
                        st.write(f"Worst case impact: {summary.get('worst_impact', 0):.2%}")
                    
                        scenarios = stress_results.get('scenarios', {})
                        for scenario_id, scenario_data in scenarios.items():
                            results = scenario_data.get('results', {})
                            st.subheader(f"Scenario Details: {scenario_type}")
                        
                            fig = go.Figure()
                            
                            fig.add_trace(go.Scatter(
                                x=data.index,
                                y=data['Close'],
                                name='Historical',
                                line=dict(color='blue')
                            ))
                            
                            if 'projected_prices' in results:
                                projected_prices = results['projected_prices']
                                projection_dates_str = results.get('projection_dates', [])
                                
                                if projected_prices and projection_dates_str:
                                    projection_dates = [pd.to_datetime(d) for d in projection_dates_str]
                                    
                                    fig.add_trace(go.Scatter(
                                        x=projection_dates,
                                        y=projected_prices,
                                        name='Projected',
                                        line=dict(color='red', dash='dash')
                                    ))
                                    
                            fig.update_layout(
                                title='Projected Price Path',
                                xaxis_title='Date',
                                yaxis_title='Price',
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Scenario analysis failed. Please check the configuration.")
                        
        except Exception as e:
            st.error(f"Error running scenario analysis: {str(e)}")

def main():
    st.set_page_config(layout="wide", page_title="AI Stock Trading System")
    st.title("AI Stock Trading System - Enhanced Risk Dashboard")

    risk_dashboard = EnhancedRiskDashboard()
    data_fetcher = DataFetcher()

    page = st.sidebar.selectbox("Choose Analysis", ["Technical Indicators", "Risk Assessment", "Scenario Analysis"])
    
    symbol = st.text_input("Enter Stock Symbol:", value="AAPL")
    
    period_options = {
        "1 Month": "1mo",
        "3 Months": "3mo", 
        "6 Months": "6mo",
        "1 Year": "1y",
        "2 Years": "2y",
        "5 Years": "5y"
    }
    
    selected_period = st.selectbox(
        "Select Time Period:",
        options=list(period_options.keys()),
        index=3
    )
    period = period_options[selected_period]

    try:
        if st.button("Analyze"):
            with st.spinner(f"Fetching data for {symbol}..."):
                data = data_fetcher.get_stock_data(symbol, period=period)
                
            if data is None or len(data) == 0:
                st.error(f"No data available for {symbol}. Please check the symbol and try again.")
                return
                
            data['symbol'] = symbol
            
            st.session_state.current_data = data

            if page == "Technical Indicators":
                st.write(f"Original Stock Data for {symbol}:")
                st.dataframe(data.tail())

                col1, col2 = st.columns(2)
                with col1:
                    period = st.number_input("Enter SMA period:", min_value=1, max_value=100, value=14)
                    if st.button("Calculate SMA"):
                        sma_indicator = SMAIndicator(period=period)
                        data_with_sma = sma_indicator.calculate(data)
                        st.write(f"Stock Data with SMA{period} for {symbol}:")
                        st.dataframe(data_with_sma.tail())

                with col2:
                    period = st.number_input("Enter RSI period:", min_value=1, max_value=100, value=14)
                    if st.button("Calculate RSI"):
                        data[f"RSI{period}"] = ta.rsi(data['Close'], length=period)
                        st.write(f"Stock Data with RSI{period} for {symbol}:")
                        st.dataframe(data.tail())

            elif page == "Risk Assessment":
                st.subheader(f"Risk Analysis for {symbol}")
                risk_dashboard.display_risk_metrics(data)

            elif page == "Scenario Analysis":
                st.subheader(f"Scenario Analysis for {symbol}")
                risk_dashboard.display_scenario_analysis(data)

    except Exception as e:
        st.error(f"Error in analysis: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Enhanced Dashboard - Sprint 2**
    
    This dashboard includes:
    - Comprehensive risk metrics
    - Multiple VaR calculation methods
    - Tail risk analysis
    - Scenario-based stress testing
    """)

if __name__ == "__main__":
    main()
