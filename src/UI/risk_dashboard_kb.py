import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from crewai import Crew, Task
from src.Agents.Risk_Assessment_Agent.risk_assessment_agent import RiskAssessmentAgent
from src.Agents.Scenario_Agents.scenario_simulation_agent import ScenarioSimulationAgent

class RiskDashboardKB:
    def __init__(self):
        self.risk_agent = RiskAssessmentAgent()
        self.scenario_agent = ScenarioSimulationAgent()

    def setup_crew(self):
        """Setup CrewAI agents for risk analysis"""
        crew = Crew(
            agents=[self.risk_agent, self.scenario_agent],
            tasks=[
                Task(
                    description="Calculate portfolio risk metrics",
                    agent=self.risk_agent
                ),
                Task(
                    description="Simulate market scenarios",
                    agent=self.scenario_agent
                )
            ]
        )
        return crew

    def plot_risk_metrics(self, portfolio_data):
        """Create risk metrics visualization using agent analysis"""
     
        risk_metrics = self.risk_agent.calculate_portfolio_risk(portfolio_data)
        
        fig = go.Figure()
        
        # Plot VaR
        fig.add_trace(go.Scatter(
            x=portfolio_data.index,
            y=[risk_metrics['var_95']] * len(portfolio_data),
            name='95% VaR',
            line=dict(color='red', dash='dash')
        ))
        
        # Plot returns
        returns = portfolio_data['Close'].pct_change()
        fig.add_trace(go.Scatter(
            x=portfolio_data.index,
            y=returns,
            name='Returns',
            mode='lines'
        ))
        
        fig.update_layout(
            title='Portfolio Risk Analysis',
            xaxis_title='Date',
            yaxis_title='Returns',
            hovermode='x unified'
        )
        
        return fig

    def plot_asset_class_risks(self, portfolio_data):
        """Create asset class risk breakdown visualization"""
        risk_by_class = self.risk_agent.analyze_asset_class_risks(portfolio_data)
        
      
        asset_classes = list(risk_by_class.keys())
        metrics = ['var_95', 'volatility']
        
        heatmap_data = []
        for asset_class in asset_classes:
            for metric in metrics:
                heatmap_data.append({
                    'Asset Class': asset_class,
                    'Metric': metric,
                    'Value': risk_by_class[asset_class][metric]
                })
                
        df = pd.DataFrame(heatmap_data)
        fig = px.heat_map(
            df, 
            x='Asset Class',
            y='Metric',
            values='Value',
            title='Risk Metrics by Asset Class'
        )
        
        return fig

    def display_dashboard(self, portfolio_data):
        """Display complete risk dashboard with agent-powered analytics"""
        st.title('Portfolio Risk Assessment Dashboard')
        
        crew = self.setup_crew()
        
        st.subheader('Risk Metrics Analysis')
        risk_fig = self.plot_risk_metrics(portfolio_data)
        st.plotly_chart(risk_fig)
        
        st.subheader('Asset Class Risk Breakdown')
        class_fig = self.plot_asset_class_risks(portfolio_data)
        st.plotly_chart(class_fig)
        
        # Validation Results
        st.subheader('Risk Metrics Validation')
        validation = self.risk_agent.validate_risk_metrics(
            self.risk_agent.calculate_portfolio_risk(portfolio_data),
            portfolio_data
        )
        st.write(validation)
        
        # Market Scenario Analysis
        st.subheader('Market Scenario Analysis')
        scenarios = self.scenario_agent.get_scenarios()
        st.write(scenarios)
