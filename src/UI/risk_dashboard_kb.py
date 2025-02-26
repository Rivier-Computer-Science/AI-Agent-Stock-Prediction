import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from crewai import Crew
from src.Agents.Risk_Assessment_Agent.risk_assessment_agent import RiskAssessmentAgent
from src.Agents.Scenario_Agents.scenario_simulation_agent import ScenarioSimulationAgent
from src.Data_Retrieval.data_fetcher import DataFetcher
import logging
import json
import time

class RiskDashboardKB:
    """
    Enhanced Risk Dashboard for Portfolio Analysis
    
    Provides comprehensive risk visualization and analysis through an
    interactive Streamlit interface powered by CrewAI agents.
    """
    
    def __init__(self):
        """Initialize the dashboard components and agents"""
        self.risk_agent = RiskAssessmentAgent()
        self.scenario_agent = ScenarioSimulationAgent()
        self.data_fetcher = DataFetcher()
        self.logger = logging.getLogger("RiskDashboard")
        
        # Initialize session state for persistence
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state for persistent storage"""
        if 'favorite_symbols' not in st.session_state:
            st.session_state.favorite_symbols = []
            
        if 'comparison_symbols' not in st.session_state:
            st.session_state.comparison_symbols = []
            
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = {}
            
        if 'current_data' not in st.session_state:
            st.session_state.current_data = None
            
        if 'theme' not in st.session_state:
            st.session_state.theme = 'light'
            
    def setup_crew(self, portfolio_data=None, parameters=None):
        """
        Setup CrewAI agents for risk analysis
        
        Args:
            portfolio_data: Portfolio data for analysis
            parameters: Custom parameters for agents
            
        Returns:
            Crew: Configured CrewAI crew
        """
        # Set portfolio data if provided
        if portfolio_data is not None:
            self.risk_agent.set_portfolio_data(portfolio_data)
            
        # Configure agents with custom parameters if provided
        if parameters is not None:
            risk_params = parameters.get('risk_agent', {})
            scenario_params = parameters.get('scenario_agent', {})
            
            # Apply risk agent parameters
            for param, value in risk_params.items():
                if hasattr(self.risk_agent, param):
                    setattr(self.risk_agent, param, value)
            
            # Apply scenario agent parameters
            for param, value in scenario_params.items():
                if hasattr(self.scenario_agent, param):
                    setattr(self.scenario_agent, param, value)
        
        # Create and configure crew
        crew = Crew(
            agents=[self.risk_agent, self.scenario_agent],
            tasks=[
                self.risk_agent.calculate_portfolio_risk(portfolio_data),
                self.scenario_agent.run_simulation()
            ]
        )
        
        return crew
        
    def display_risk_metrics(self, data, confidence_level=0.95, volatility_window=30):
        """
        Display comprehensive risk metrics visualizations
        
        Args:
            data: Portfolio data for visualization
            confidence_level: Confidence level for VaR calculations
            volatility_window: Window size for volatility calculations
        """
        try:
            # Store data in session state
            st.session_state.current_data = data
            
            # Symbol info
            symbol = data.get('symbol', 'Unknown')
            
            # Calculate risk metrics
            with st.spinner("Calculating risk metrics..."):
                self.risk_agent.set_portfolio_data(data)
                risk_metrics = self.risk_agent.get_risk_metrics()
                
                # Store in session state
                st.session_state.analysis_results[symbol] = risk_metrics
                
            # Check if metrics were calculated successfully
            if not risk_metrics:
                st.error("Failed to calculate risk metrics. Please check the data.")
                return
                
            # Create tabs for different visualizations
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "VaR Analysis", 
                "Drawdown", 
                "Volatility", 
                "Tail Risk",
                "Validation"
            ])
            
            with tab1:
                self._display_var_analysis(data, risk_metrics, confidence_level)
                
            with tab2:
                self._display_drawdown_analysis(data, risk_metrics)
                
            with tab3:
                self._display_volatility_analysis(data, risk_metrics, volatility_window)
                
            with tab4:
                self._display_tail_risk_analysis(data, risk_metrics)
                
            with tab5:
                self._display_validation_results(risk_metrics)
                
            # Summary metrics table
            st.subheader('Risk Metrics Summary')
            self._display_metrics_summary(risk_metrics, confidence_level, volatility_window)
            
            # Risk recommendations
            st.subheader('Risk Management Recommendations')
            stress_results = self.risk_agent.run_stress_tests()
            recommendations = self.risk_agent.get_risk_recommendations(risk_metrics, stress_results)
            
            if recommendations:
                warning_level = recommendations.get('warning_level', 'Low')
                warning_color = {
                    'Low': 'green',
                    'Medium': 'orange',
                    'High': 'red'
                }.get(warning_level, 'blue')
                
                st.markdown(f"**Warning Level:** <span style='color:{warning_color}'>{warning_level}</span>", 
                            unsafe_allow_html=True)
                
                for i, rec in enumerate(recommendations.get('recommendations', [])):
                    st.markdown(f"**{i+1}.** {rec}")
            
        except Exception as e:
            self.logger.error(f"Error displaying risk metrics: {str(e)}")
            st.error(f"Error displaying risk metrics: {str(e)}")
            
    def _display_var_analysis(self, data, risk_metrics, confidence_level):
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
                annotation_text=f"Historical VaR ({confidence_level*100:.0f}%): {var_hist:.2%}",
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
                annotation_text=f"CVaR ({confidence_level*100:.0f}%): {cvar:.2%}",
                annotation_position="bottom left"
            )
        
        fig.update_layout(
            title=f'Value at Risk Analysis ({confidence_level*100:.0f}% Confidence)',
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
        
        # Add annotations for major drawdowns
        if recovery_periods:
            for period in recovery_periods:
                if period.get('max_drawdown', 0) <= -0.1:  # Only annotate significant drawdowns
                    fig.add_vline(
                        x=period.get('start_date'),
                        line_width=1,
                        line_dash="dot",
                        line_color="gray"
                    )
                    
                    fig.add_annotation(
                        x=period.get('start_date'),
                        y=period.get('max_drawdown'),
                        text=f"{period.get('max_drawdown', 0):.1%}",
                        showarrow=True,
                        arrowhead=1
                    )
        
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
            
    def _display_volatility_analysis(self, data, risk_metrics, window):
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
            name=f'{window}-Day Volatility',
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
        
        # Add volatility regime lines
        low_vol = 0.15  # 15% annualized
        high_vol = 0.30  # 30% annualized
        
        fig.add_shape(
            type="line",
            x0=volatility.index[0],
            x1=volatility.index[-1],
            y0=low_vol,
            y1=low_vol,
            line=dict(color="green", width=1, dash="dash")
        )
        
        fig.add_shape(
            type="line",
            x0=volatility.index[0],
            x1=volatility.index[-1],
            y0=high_vol,
            y1=high_vol,
            line=dict(color="red", width=1, dash="dash")
        )
        
        # Add annotations for volatility regimes
        fig.add_annotation(
            x=volatility.index[-1],
            y=low_vol,
            text="Low Volatility",
            showarrow=False,
            yshift=10
        )
        
        fig.add_annotation(
            x=volatility.index[-1],
            y=high_vol,
            text="High Volatility",
            showarrow=False,
            yshift=10
        )
        
        fig.update_layout(
            title=f'Annualized Rolling Volatility Analysis',
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
            
            **Volatility regimes** help categorize market conditions:
            - **Low volatility** (under 15%): Typically calm, trending markets
            - **Medium volatility** (15-30%): Normal market conditions with moderate price swings
            - **High volatility** (over 30%): Turbulent markets with large price movements, often 
              associated with crises or major economic events
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
        x = np.linspace(mean - 4*std, mean + 4*std, 100)
        y = stats.norm.pdf(x, mean, std)
        
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            name='Normal Distribution',
            line=dict(color='rgba(54, 162, 235, 1)')
        ))
        
        # Add annotations for tail risk metrics
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
        
        fig.add_annotation(
            x=0.05,
            y=0.85,
            xref="paper",
            yref="paper",
            text=f"Tail Risk Score: {tail_risk_score:.4f}",
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
            
            - **Skewness**: Measures the asymmetry of returns. Negative skewness (< 0) indicates 
              a distribution with a longer left tail, meaning more frequent small gains but occasional 
              large losses - a common pattern in financial markets.
              
            - **Kurtosis**: Measures the "fatness" of the tails. Higher kurtosis (> 3) indicates 
              more frequent extreme values than a normal distribution would predict.
              
            - **Tail Risk Score**: A composite metric combining skewness and kurtosis to quantify 
              overall tail risk exposure.
            """)
            
    def _display_validation_results(self, risk_metrics):
        """Display risk metrics validation results"""
        st.subheader('Risk Metrics Validation')
        
        validation = risk_metrics.get('validation')
        
        if validation is None:
            st.warning("Validation data not available")
            return
            
        # Show validation issues and warnings
        issues = validation.get('issues', [])
        warnings = validation.get('warnings', [])
        validations_passed = validation.get('validations_passed', False)
        
        # Status indicator
        status_color = "green" if validations_passed else "red"
        status_text = "PASSED" if validations_passed else "FAILED"
        
        st.markdown(
            f"<div style='background-color:{status_color}; padding:10px; border-radius:5px;'>"
            f"<h3 style='color:white; text-align:center; margin:0;'>Validation {status_text}</h3>"
            f"</div>",
            unsafe_allow_html=True
        )
        
        # Display issues
        if issues:
            st.subheader("Validation Issues")
            for issue in issues:
                st.markdown(f"- 🔴 {issue}")
        
        # Display warnings
        if warnings:
            st.subheader("Validation Warnings")
            for warning in warnings:
                st.markdown(f"- 🟠 {warning}")
                
        # If all validations passed and no warnings
        if validations_passed and not issues and not warnings:
            st.success("All risk metrics passed validation with no warnings.")
            
        # Explanation
        with st.expander("About Risk Metric Validation"):
            st.write("""
            **Risk metric validation** ensures that calculated risk measures are reliable and 
            consistent. The validation process checks for:
            
            - **Consistency**: Ensuring different calculation methods produce similar results
            - **Reasonableness**: Verifying metrics are within expected ranges
            - **Logical relationships**: Confirming relationships between metrics (e.g., CVaR should be 
              more extreme than VaR)
            
            Issues are critical problems that may indicate calculation errors, while warnings highlight 
            potential concerns that require attention but don't invalidate the metrics.
            """)
            
    def _display_metrics_summary(self, risk_metrics, confidence_level, volatility_window):
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
                f'VaR ({confidence_level*100:.0f}%)',
                f'CVaR ({confidence_level*100:.0f}%)',
                'Max Drawdown',
                f'Current Volatility ({volatility_window}-day)',
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
        
        # Add interpretation column
        interpretation = []
        
        # VaR interpretation
        if var_95 is not None:
            if var_95 > -0.01:
                interpretation.append("Low risk")
            elif var_95 > -0.02:
                interpretation.append("Moderate risk")
            else:
                interpretation.append("High risk")
        else:
            interpretation.append("N/A")
            
        # CVaR interpretation
        if cvar_95 is not None:
            if cvar_95 > -0.015:
                interpretation.append("Low tail risk")
            elif cvar_95 > -0.03:
                interpretation.append("Moderate tail risk")
            else:
                interpretation.append("High tail risk")
        else:
            interpretation.append("N/A")
            
        # Max drawdown interpretation
        if max_dd is not None:
            if max_dd > -0.1:
                interpretation.append("Low drawdown risk")
            elif max_dd > -0.2:
                interpretation.append("Moderate drawdown risk")
            else:
                interpretation.append("High drawdown risk")
        else:
            interpretation.append("N/A")
            
        # Volatility interpretation
        if current_vol is not None:
            if current_vol < 0.15:
                interpretation.append("Low volatility")
            elif current_vol < 0.30:
                interpretation.append("Moderate volatility")
            else:
                interpretation.append("High volatility")
        else:
            interpretation.append("N/A")
            
        # Skewness interpretation
        if 'skewness' in tail_risk:
            skew = tail_risk['skewness']
            if skew < -0.5:
                interpretation.append("Negatively skewed (large losses)")
            elif skew > 0.5:
                interpretation.append("Positively skewed (large gains)")
            else:
                interpretation.append("Near symmetric")
        else:
            interpretation.append("N/A")
            
        # Kurtosis interpretation
        if 'kurtosis' in tail_risk:
            kurt = tail_risk['kurtosis']
            if kurt > 5:
                interpretation.append("Fat tails (extreme events likely)")
            elif kurt > 3:
                interpretation.append("Moderately fat tails")
            else:
                interpretation.append("Near normal tails")
        else:
            interpretation.append("N/A")
            
        # Tail risk score interpretation
        if 'tail_risk_score' in tail_risk:
            score = tail_risk['tail_risk_score']
            if score < 0.5:
                interpretation.append("Low tail risk exposure")
            elif score < 1.0:
                interpretation.append("Moderate tail risk exposure")
            else:
                interpretation.append("High tail risk exposure")
        else:
            interpretation.append("N/A")
            
        # Add interpretation to summary data
        summary_data['Interpretation'] = interpretation
        
        # Display as table
        st.table(pd.DataFrame(summary_data))
        
    def display_asset_class_breakdown(self, portfolio_data, asset_class_mapping=None):
        """
        Display asset class risk breakdown
        
        Args:
            portfolio_data: Portfolio data for analysis
            asset_class_mapping: Dictionary mapping symbols to asset classes
        """
        st.subheader("Asset Class Risk Breakdown")
        
        try:
            # Configure risk agent
            self.risk_agent.set_portfolio_data(portfolio_data)
            
            if asset_class_mapping:
                self.risk_agent.set_asset_class_mapping(asset_class_mapping)
                
            # Calculate asset class risks
            with st.spinner("Calculating asset class risks..."):
                asset_risks = self.risk_agent.analyze_asset_risks(portfolio_data, asset_class_mapping)
                
            if not asset_risks:
                st.warning("Unable to calculate asset class risks. This feature requires multi-asset portfolio data.")
                return
                
            # Remove portfolio-level metrics for visualization
            if '_portfolio' in asset_risks:
                portfolio_metrics = asset_risks.pop('_portfolio')
            else:
                portfolio_metrics = None
                
            # Convert to DataFrame for visualization
            risk_data = []
            
            for asset_class, metrics in asset_risks.items():
                risk_data.append({
                    'Asset Class': asset_class,
                    'Weight': metrics.get('weight_normalized', 0) * 100,
                    'VaR (95%)': metrics.get('var_95', 0) * 100,
                    'Volatility': metrics.get('volatility', 0) * 100,
                    'Risk Contribution': metrics.get('volatility_contribution', 0) * 100
                })
                
            if not risk_data:
                st.warning("No asset class risk data available.")
                return
                
            df_risks = pd.DataFrame(risk_data)
            
            # Create tabs for different visualizations
            tab1, tab2 = st.tabs(["Risk Charts", "Heatmap"])
            
            with tab1:
                # Create bar charts
                fig1 = px.bar(
                    df_risks,
                    x='Asset Class',
                    y=['Weight', 'Risk Contribution'],
                    title='Asset Class Weights vs. Risk Contribution',
                    barmode='group'
                )
                
                st.plotly_chart(fig1, use_container_width=True)
                
                fig2 = px.bar(
                    df_risks,
                    x='Asset Class',
                    y=['VaR (95%)', 'Volatility'],
                    title='VaR and Volatility by Asset Class',
                    barmode='group'
                )
                
                # Adjust y-axis to display negative values properly
                fig2.update_layout(yaxis_title='Percentage (%)')
                
                st.plotly_chart(fig2, use_container_width=True)
                
            with tab2:
                # Create heatmap
                heatmap_data = df_risks.set_index('Asset Class')
                
                fig3 = px.imshow(
                    heatmap_data,
                    labels=dict(x="Metric", y="Asset Class", color="Value (%)"),
                    x=heatmap_data.columns,
                    y=heatmap_data.index,
                    color_continuous_scale='Blues_r',  # Reversed blues (darker = higher absolute value)
                    title='Asset Class Risk Metrics Heatmap'
                )
                
                st.plotly_chart(fig3, use_container_width=True)
                
            # Display metrics table
            st.subheader("Asset Class Risk Metrics")
            
            # Format for display
            display_df = df_risks.copy()
            for col in display_df.columns:
                if col != 'Asset Class':
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%")
                    
            st.table(display_df)
            
            # Explanation
            with st.expander("Understanding Asset Class Risk Breakdown"):
                st.write("""
                **Asset Class Risk Breakdown** provides insight into how different asset classes 
                contribute to your portfolio's overall risk profile:
                
                - **Weight**: The percentage allocation of each asset class in the portfolio
                - **Risk Contribution**: How much each asset class contributes to the portfolio's 
                  overall risk, which may differ from its weight due to volatility and correlations
                - **VaR (95%)**: The Value at Risk for each asset class individually
                - **Volatility**: The annualized standard deviation of returns for each asset class
                
                *Note: Asset classes with high risk contribution relative to their weight may 
                warrant closer examination for potential portfolio optimization.*
                """)
                
        except Exception as e:
            self.logger.error(f"Error displaying asset class breakdown: {str(e)}")
            st.error(f"Error displaying asset class breakdown: {str(e)}")
            
    def display_scenario_analysis(self, portfolio_data):
        """
        Display scenario analysis using CrewAI agents
        
        Args:
            portfolio_data: Portfolio data for scenario analysis
        """
        st.subheader("Scenario Analysis")
        
        try:
            # Configure risk agent
            self.risk_agent.set_portfolio_data(portfolio_data)
            
            # Scenario selection
            scenario_type = st.selectbox(
                "Select Scenario Type",
                ["Historical Crisis", "Market Shock", "Interest Rate Change", "Monte Carlo Simulation", "Custom"]
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
                correlation_adjustment = st.slider("Correlation Adjustment", min_value=-0.5, max_value=0.5, value=0.2, step=0.1)
                
                params = {"shock_size": shock_size/100, "correlation_adjustment": correlation_adjustment}
                scenario_config = {"type": "custom", "params": params}
                
            elif scenario_type == "Interest Rate Change":
                rate_change = st.slider("Interest Rate Change (bps)", min_value=-200, max_value=200, value=50, step=25)
                
                # Convert to scenario params (simplified mapping to custom scenario)
                # In a real implementation, this would use a specialized interest rate model
                shock_size = -0.05 if rate_change > 0 else 0.03  # Simplified impact
                
                params = {"shock_size": shock_size, "correlation_adjustment": 0.1, "rate_change": rate_change/10000}
                scenario_config = {"type": "custom", "params": params}
                
            elif scenario_type == "Monte Carlo Simulation":
                distribution = st.selectbox("Distribution Type", ["normal", "t", "skewed"])
                num_simulations = st.slider("Number of Simulations", min_value=100, max_value=10000, value=1000, step=100)
                
                params = {"distribution": distribution, "num_simulations": num_simulations}
                scenario_config = {"type": "monte_carlo", "params": params}
                
            else:  # Custom
                shock_size = st.slider("Price Shock (%)", min_value=-50, max_value=50, value=-15, step=5)
                vol_change = st.slider("Volatility Change (%)", min_value=-50, max_value=200, value=50, step=10)
                correlation_change = st.slider("Correlation Change", min_value=-0.5, max_value=0.5, value=0.2, step=0.1)
                
                params = {
                    "shock_size": shock_size/100,
                    "vol_change": vol_change/100,
                    "correlation_adjustment": correlation_change
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
                    
                    # Run scenario analysis
                    scenarios = [scenario_config]
                    stress_results = self.risk_agent.run_stress_tests(scenarios)
                    
                    if stress_results:
                        self._display_scenario_results(stress_results, portfolio_data)
                    else:
                        st.error("Scenario analysis failed. Please check the configuration and try again.")
                        
        except Exception as e:
            self.logger.error(f"Error running scenario analysis: {str(e)}")
            st.error(f"Error running scenario analysis: {str(e)}")
            
    def _display_scenario_results(self, stress_results, original_data):
        """
        Display scenario analysis results
        
        Args:
            stress_results: Results from stress testing
            original_data: Original portfolio data for comparison
        """
        # Get scenarios results
        scenarios = stress_results.get('scenarios', {})
        summary = stress_results.get('summary', {})
        
        if not scenarios:
            st.error("No scenario results available.")
            return
            
        # Display summary metrics
        st.subheader("Scenario Analysis Summary")
        st.write(f"Analyzed {summary.get('num_scenarios', 0)} scenario(s)")
        st.write(f"Worst case impact: {summary.get('worst_impact', 0):.2%}")
        
        # Create tabs for each scenario
        for scenario_id, scenario_data in scenarios.items():
            configuration = scenario_data.get('configuration', {})
            results = scenario_data.get('results', {})
            
            scenario_type = results.get('stress_test_type', 'unknown')
            
            # Create expander for this scenario
            with st.expander(f"Scenario: {scenario_type.title()}", expanded=True):
                # Show configuration
                st.subheader("Scenario Configuration")
                if scenario_type == "historical":
                    st.write(f"Historical period: {results.get('period', 'Unknown')}")
                    st.write(f"Historical maximum drawdown: {results.get('max_drawdown', 0):.2%}")
                    st.write(f"Historical duration: {results.get('duration_days', 0)} days")
                elif scenario_type == "monte_carlo":
                    st.write(f"Distribution: {results.get('distribution', 'normal')}")
                    st.write(f"Simulations: {results.get('num_simulations', 0)}")
                    st.write(f"Horizon: {results.get('horizon_days', 0)} days")
                elif scenario_type == "custom":
                    st.write(f"Shock size: {results.get('shock_size', 0):.2%}")
                    st.write(f"Volatility change: {results.get('volatility_change', 0):.2%}")
                    
                # Projected price impact
                st.subheader("Projected Impact")
                
                # Create price projection chart
                fig = go.Figure()
                
                # Add historical prices
                fig.add_trace(go.Scatter(
                    x=original_data.index,
                    y=original_data['Close'],
                    name='Historical',
                    line=dict(color='blue')
                ))
                
                # Add projected prices based on scenario type
                if scenario_type == "historical" or scenario_type == "custom":
                    # Get projection data
                    recovery_path = results.get('recovery_path', []) or results.get('projected_prices', [])
                    projection_dates_str = results.get('projection_dates', [])
                    
                    if recovery_path and projection_dates_str:
                        # Convert date strings to datetime
                        projection_dates = [pd.to_datetime(d) for d in projection_dates_str]
                        
                        fig.add_trace(go.Scatter(
                            x=projection_dates,
                            y=recovery_path,
                            name='Projected',
                            line=dict(color='red', dash='dash')
                        ))
                        
                elif scenario_type == "monte_carlo":
                    # Get worst case and median case
                    worst_case = results.get('worst_case', {})
                    median_case = results.get('median_case', {})
                    projection_dates_str = results.get('projection_dates', [])
                    
                    if worst_case and median_case and projection_dates_str:
                        # Convert date strings to datetime
                        projection_dates = [pd.to_datetime(d) for d in projection_dates_str]
                        
                        # Add worst case
                        fig.add_trace(go.Scatter(
                            x=projection_dates,
                            y=worst_case.get('path', []),
                            name='Worst Case (1%)',
                            line=dict(color='red', dash='dash')
                        ))
                        
                        # Add median case
                        fig.add_trace(go.Scatter(
                            x=projection_dates,
                            y=median_case.get('path', []),
                            name='Median Case',
                            line=dict(color='orange', dash='dash')
                        ))
                        
                fig.update_layout(
                    title='Projected Price Path',
                    xaxis_title='Date',
                    yaxis_title='Price',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display risk changes
                st.subheader("Risk Metric Changes")
                
                # Create metrics based on scenario type
                if scenario_type == "historical":
                    metrics_data = {
                        'Metric': ['Current Value', 'Scenario Value', 'Impact'],
                        'Value': [
                            f"${results.get('current_value', 0):.2f}",
                            f"${results.get('scenario_value', 0):.2f}",
                            f"{results.get('value_impact', 0):.2%}"
                        ]
                    }
                    
                elif scenario_type == "monte_carlo":
                    percentiles = results.get('percentiles', {})
                    
                    metrics_data = {
                        'Percentile': ['1%', '5%', '10%', '50% (Median)', '90%', '95%', '99%'],
                        'Projected Value': [
                            f"${percentiles.get('p1', 0):.2f}",
                            f"${percentiles.get('p5', 0):.2f}",
                            f"${percentiles.get('p10', 0):.2f}",
                            f"${percentiles.get('p50', 0):.2f}",
                            f"${percentiles.get('p90', 0):.2f}",
                            f"${percentiles.get('p95', 0):.2f}",
                            f"${percentiles.get('p99', 0):.2f}"
                        ],
                        'Return': [
                            f"{(percentiles.get('p1', 0) / results.get('current_price', 1)) - 1:.2%}",
                            f"{(percentiles.get('p5', 0) / results.get('current_price', 1)) - 1:.2%}",
                            f"{(percentiles.get('p10', 0) / results.get('current_price', 1)) - 1:.2%}",
                            f"{(percentiles.get('p50', 0) / results.get('current_price', 1)) - 1:.2%}",
                            f"{(percentiles.get('p90', 0) / results.get('current_price', 1)) - 1:.2%}",
                            f"{(percentiles.get('p95', 0) / results.get('current_price', 1)) - 1:.2%}",
                            f"{(percentiles.get('p99', 0) / results.get('current_price', 1)) - 1:.2%}"
                        ]
                    }
                    
                elif scenario_type == "custom":
                    metrics_data = {
                        'Metric': ['Current Price', 'Shocked Price', 'Price Change',
                                  'Current Volatility', 'Shocked Volatility', 'Volatility Change',
                                  'Current VaR', 'Shocked VaR', 'VaR Change'],
                        'Value': [
                            f"${results.get('current_price', 0):.2f}",
                            f"${results.get('shocked_price', 0):.2f}",
                            f"{results.get('price_change', 0):.2%}",
                            f"{results.get('current_volatility', 0):.2%}",
                            f"{results.get('shocked_volatility', 0):.2%}",
                            f"{results.get('volatility_change', 0):.2%}",
                            f"{results.get('current_var', 0):.2%}",
                            f"{results.get('shocked_var', 0):.2%}",
                            f"{results.get('var_change', 0):.2%}"
                        ]
                    }
                
                # Display metrics
                st.table(pd.DataFrame(metrics_data))
                
                # Get recommendations
                recommendations = self.risk_agent.get_risk_recommendations(stress_results=stress_results)
                
                if recommendations:
                    st.subheader("Recommendations")
                    for i, rec in enumerate(recommendations.get('recommendations', [])):
                        st.markdown(f"**{i+1}.** {rec}")
    
    def display_dashboard(self, symbol=None, period="1y"):
        """
        Display complete risk dashboard with agent-powered analytics
        
        Args:
            symbol: Stock symbol to analyze (None for user input)
            period: Time period for analysis
        """
        st.title('Portfolio Risk Assessment Dashboard')
        
        # Get symbol input if not provided
        if symbol is None:
            symbol = st.text_input("Enter Stock Symbol:", value="AAPL")
        
        # Time period selection
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
            index=list(period_options.keys()).index("1 Year") if period == "1y" else 0
        )
        
        # Map to Yahoo Finance period format
        period = period_options[selected_period]
        
        # Fetch data button
        if st.button("Analyze"):
            try:
                # Fetch data
                with st.spinner(f"Fetching data for {symbol}..."):
                    data = self.data_fetcher.get_stock_data(symbol, period=period)
                    
                if data is None or len(data) == 0:
                    st.error(f"No data available for {symbol}. Please check the symbol and try again.")
                    return
                    
                # Add symbol for reference
                data['symbol'] = symbol
                
                # Store in session state
                st.session_state.current_data = data
                
                # Display tabs for different analysis
                tab1, tab2, tab3 = st.tabs(["Risk Metrics", "Asset Class Analysis", "Scenario Analysis"])
                
                with tab1:
                    # Risk metrics visualization
                    self.display_risk_metrics(data)
                    
                with tab2:
                    # Asset class breakdown (simplified for single stock)
                    asset_mapping = {symbol: "Equities"}
                    self.display_asset_class_breakdown(data, asset_mapping)
                    
                with tab3:
                    # Scenario analysis
                    self.display_scenario_analysis(data)
                    
            except Exception as e:
                self.logger.error(f"Error in dashboard: {str(e)}")
                st.error(f"Error in analysis: {str(e)}")
                
    def export_analysis_results(self, format="csv"):
        """
        Export analysis results to file
        
        Args:
            format: Export format ("csv" or "json")
        """
        if not st.session_state.analysis_results:
            st.warning("No analysis results available for export.")
            return
            
        try:
            # Convert results to desired format
            if format == "csv":
                # Create DataFrame from results
                export_data = {}
                
                for symbol, metrics in st.session_state.analysis_results.items():
                    # Handle scalar values
                    for metric_name in ['var_95', 'var_99', 'cvar_95', 'cvar_99']:
                        if metric_name in metrics:
                            export_data[f"{symbol}_{metric_name}"] = [metrics[metric_name]]
                    
                    # Handle tail risk dictionary
                    if 'tail_risk' in metrics:
                        for key, value in metrics['tail_risk'].items():
                            export_data[f"{symbol}_tail_risk_{key}"] = [value]
                
                df_export = pd.DataFrame(export_data)
                
                # Convert to CSV
                csv_data = df_export.to_csv(index=False)
                
                # Create download button
                st.download_button(
                    label=f"Download Results (CSV)",
                    data=csv_data,
                    file_name="risk_analysis_results.csv",
                    mime="text/csv"
                )
                
            elif format == "json":
                # Convert to JSON
                import json
                
                # Create serializable version of results
                serializable_results = {}
                
                for symbol, metrics in st.session_state.analysis_results.items():
                    serializable_metrics = {}
                    
                    # Handle scalar values
                    for metric_name in ['var_95', 'var_99', 'cvar_95', 'cvar_99']:
                        if metric_name in metrics:
                            serializable_metrics[metric_name] = float(metrics[metric_name])
                    
                    # Handle tail risk dictionary
                    if 'tail_risk' in metrics:
                        serializable_metrics['tail_risk'] = {
                            k: float(v) for k, v in metrics['tail_risk'].items()
                        }
                    
                    serializable_results[symbol] = serializable_metrics
                
                json_data = json.dumps(serializable_results, indent=2)
                
                # Create download button
                st.download_button(
                    label=f"Download Results (JSON)",
                    data=json_data,
                    file_name="risk_analysis_results.json",
                    mime="application/json"
                )
                
        except Exception as e:
            self.logger.error(f"Error exporting analysis results: {str(e)}")
            st.error(f"Error exporting results: {str(e)}")
            
    def main(self):
        """Main dashboard application"""
        st.set_page_config(
            page_title="AI-Powered Risk Assessment Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Display sidebar
        with st.sidebar:
            st.title("Risk Dashboard Controls")
            
            # User preferences
            st.subheader("User Preferences")
            
            # Theme selection
            theme = st.selectbox(
                "Select Theme",
                ["Light", "Dark"],
                index=0 if st.session_state.theme == 'light' else 1
            )
            st.session_state.theme = theme.lower()
            
            # Apply theme
            if st.session_state.theme == 'dark':
                st.markdown("""
                    <style>
                    .stApp {
                        background-color: #1E1E1E;
                        color: white;
                    }
                    </style>
                """, unsafe_allow_html=True)
            
            # Favorite symbols
            st.subheader("Favorite Symbols")
            new_symbol = st.text_input("Add Symbol to Favorites")
            
            if st.button("Add to Favorites") and new_symbol:
                if new_symbol not in st.session_state.favorite_symbols:
                    st.session_state.favorite_symbols.append(new_symbol)
                    st.success(f"Added {new_symbol} to favorites")
                else:
                    st.info(f"{new_symbol} is already in favorites")
            
            # Display favorite symbols
            if st.session_state.favorite_symbols:
                selected_favorite = st.selectbox(
                    "Select from Favorites",
                    [""] + st.session_state.favorite_symbols
                )
                
                if selected_favorite and st.button(f"Analyze {selected_favorite}"):
                    # Switch to main content area
                    self.display_dashboard(symbol=selected_favorite)
                    
            # Export options
            if st.session_state.analysis_results:
                st.subheader("Export Options")
                export_format = st.selectbox("Export Format", ["CSV", "JSON"])
                
                if st.button("Export Results"):
                    self.export_analysis_results(format=export_format.lower())
            
            # Advanced settings
            with st.expander("Advanced Settings"):
                confidence_level = st.slider(
                    "VaR Confidence Level", 
                    min_value=0.9, 
                    max_value=0.99, 
                    value=0.95,
                    step=0.01
                )
                
                volatility_window = st.slider(
                    "Volatility Window (days)", 
                    min_value=5, 
                    max_value=60, 
                    value=30,
                    step=5
                )
                
                # Store in session state
                st.session_state.confidence_level = confidence_level
                st.session_state.volatility_window = volatility_window
        
        # Main content
        self.display_dashboard()
        
        # Footer
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: gray;'>"
            "AI-Powered Risk Assessment Dashboard | Created with CrewAI &amp; Streamlit"
            "</div>", 
            unsafe_allow_html=True
        )
