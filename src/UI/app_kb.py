import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

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

class RiskAssessmentDashboard:
    def __init__(self):
        self.risk_agent = RiskAssessmentAgent()
        self.scenario_agent = ScenarioSimulationAgent()
        
    def setup_crew(self):
        return Crew(
            agents=[self.risk_agent, self.scenario_agent],
            tasks=[
                self.risk_agent.calculate_portfolio_risk(None),
                self.scenario_agent.run_simulation()
            ]
        )
        
    def display_risk_metrics(self, data):
        risk_metrics = RiskMetricsKB(data=data)
        metrics = risk_metrics.analyze_risk_metrics()
        
        if metrics:
            # VaR Analysis
            col1, col2 = st.columns(2)
            with col1:
                fig_var = go.Figure()
                fig_var.add_trace(go.Scatter(
                    y=[metrics['var_95']], 
                    name='95% VaR',
                    line=dict(color='red')
                ))
                fig_var.update_layout(title='Value at Risk Analysis')
                st.plotly_chart(fig_var)
            
            with col2:
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(
                    x=metrics['drawdown'].index,
                    y=metrics['drawdown'].values,
                    name='Drawdown',
                    fill='tonexty'
                ))
                fig_dd.update_layout(title='Historical Drawdown')
                st.plotly_chart(fig_dd)
            
            # Volatility Analysis
            st.subheader('Volatility Analysis')
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(
                x=metrics['volatility'].index,
                y=metrics['volatility'].values,
                name='Rolling Volatility'
            ))
            fig_vol.update_layout(title='30-Day Rolling Volatility')
            st.plotly_chart(fig_vol)
            
            st.subheader('Risk Metrics Summary')
            summary_data = {
                'Metric': ['VaR (95%)', 'Max Drawdown', 'Current Volatility'],
                'Value': [
                    f"{metrics['var_95']:.2%}",
                    f"{metrics['drawdown'].min():.2%}",
                    f"{metrics['volatility'][-1]:.2%}"
                ]
            }
            st.table(pd.DataFrame(summary_data))

def main():
    st.set_page_config(layout="wide", page_title="AI Stock Trading System")
    st.title("AI Stock Trading System")

    risk_dashboard = RiskAssessmentDashboard()
    data_fetcher = DataFetcher()

    page = st.sidebar.selectbox("Choose Analysis", ["Technical Indicators", "Risk Assessment"])
    symbol = st.text_input("Enter Stock Symbol:", value="AAPL")

    try:
        # Fetch data
        data = data_fetcher.get_stock_data(symbol)

        if page == "Technical Indicators":
            st.write(f"Original Stock Data for {symbol}:")
            st.dataframe(data.tail())

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Calculate SMA"):
                    period = st.number_input("Enter SMA period:", 
                                           min_value=1, max_value=100, value=14)
                    sma_indicator = SMAIndicator(period=period)
                    data_with_sma = sma_indicator.calculate(data)
                    st.write(f"Stock Data with SMA{period} for {symbol}:")
                    st.dataframe(data_with_sma.tail())

            with col2:
                if st.button("Calculate RSI"):
                    period = st.number_input("Enter RSI period:", 
                                           min_value=1, max_value=100, value=14)
                    data[f"RSI{period}"] = ta.rsi(data['Close'], length=period)
                    st.write(f"Stock Data with RSI{period} for {symbol}:")
                    st.dataframe(data.tail())

        elif page == "Risk Assessment":
            st.subheader(f"Risk Analysis for {symbol}")
            risk_dashboard.display_risk_metrics(data)

            st.subheader("Scenario Analysis")
            if st.button("Run Scenario Analysis"):
                crew = risk_dashboard.setup_crew()
                scenario_results = crew.kickoff()
                st.write(scenario_results)

        if st.button("Fetch Latest Data"):
            latest_data = data_fetcher.get_stock_data(symbol)
            st.write(f"Latest Stock Data for {symbol}:")
            st.dataframe(latest_data.tail())

    except Exception as e:
        st.error(f"Error in analysis: {str(e)}")

if __name__ == "__main__":
    main()
