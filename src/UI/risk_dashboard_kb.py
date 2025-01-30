import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class RiskDashboard:
    def __init__(self):
        self.update_interval = 60  # update interval in seconds

    def create_filters(self):
        """Create filtering options for the dashboard"""
        st.sidebar.subheader("Filter Options")
        

        asset_classes = ["Stocks", "Bonds", "Commodities", "Forex"]
        selected_assets = st.sidebar.multiselect(
            "Select Asset Classes",
            asset_classes,
            default=asset_classes
        )
        
   
        time_periods = {
            "1D": 1,
            "1W": 7,
            "1M": 30,
            "3M": 90,
            "6M": 180,
            "1Y": 365
        }
        selected_period = st.sidebar.selectbox(
            "Select Time Period",
            list(time_periods.keys())
        )
        
        return selected_assets, time_periods[selected_period]

    def plot_var_chart(self, var_value, returns, confidence_levels=[0.95, 0.99]):
        """Create interactive VaR visualization with multiple confidence levels"""
        fig = go.Figure()
        
        
        fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=50,
            name='Returns Distribution',
            hovertemplate="Return: %{x:.2%}<br>Count: %{y}"
        ))
        
        # Add VaR lines for different confidence levels
        colors = ['red', 'orange']
        for conf, color in zip(confidence_levels, colors):
            var = np.percentile(returns, (1 - conf) * 100)
            fig.add_vline(
                x=var,
                line_color=color,
                line_dash='dash',
                annotation_text=f'{conf*100}% VaR: {var:.2%}'
            )
        
        fig.update_layout(
            title='Value at Risk Distribution',
            xaxis_title='Returns',
            yaxis_title='Frequency',
            hovermode='x unified'
        )
        return fig

    def plot_drawdown_chart(self, drawdown_series, rolling_windows=[30, 90]):
        """Create interactive drawdown visualization with moving averages"""
        fig = go.Figure()
        
       
        fig.add_trace(go.Scatter(
            x=drawdown_series.index,
            y=drawdown_series.values,
            mode='lines',
            name='Drawdown',
            hovertemplate="Date: %{x}<br>Drawdown: %{y:.2%}"
        ))
        
       
        for window in rolling_windows:
            ma = drawdown_series.rolling(window=window).mean()
            fig.add_trace(go.Scatter(
                x=ma.index,
                y=ma.values,
                mode='lines',
                line={'dash': 'dot'},
                name=f'{window}D MA',
                visible='legendonly'
            ))
        
        fig.update_layout(
            title='Portfolio Drawdown',
            xaxis_title='Date',
            yaxis_title='Drawdown %',
            hovermode='x unified'
        )
        return fig

    def display_risk_metrics(self, risk_data):
        """Display interactive risk metrics dashboard"""
        # Create filters
        selected_assets, time_period = self.create_filters()
        

        filtered_data = self.filter_risk_data(risk_data, selected_assets, time_period)
        
        st.subheader('Risk Metrics Dashboard')
        
 
        auto_refresh = st.checkbox("Enable Auto-refresh")
        if auto_refresh:
            st.empty()  
            
        # Display visualizations
        st.plotly_chart(self.plot_var_chart(
            filtered_data['var'],
            filtered_data['returns']
        ))
        
        st.plotly_chart(self.plot_drawdown_chart(
            filtered_data['drawdown']
        ))
        
        if 'correlation' in filtered_data:
            st.plotly_chart(self.plot_correlation_heatmap(
                filtered_data['correlation']
            ))
        
        self.display_risk_summary(filtered_data)
        
     
        if auto_refresh:
            st.experimental_rerun()

    def filter_risk_data(self, risk_data, selected_assets, time_period):
        """Filter risk data based on selected criteria"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=time_period)
        
        filtered_data = {
            'returns': risk_data['returns'][start_date:],
            'drawdown': risk_data['drawdown'][start_date:],
            'volatility': risk_data['volatility'][start_date:],
        }
        
        if 'correlation' in risk_data:
            filtered_data['correlation'] = risk_data['correlation'].loc[selected_assets, selected_assets]
        
        # Recalculate VaR for filtered data
        filtered_data['var'] = np.percentile(filtered_data['returns'], 5)
        
        return filtered_data

    def display_risk_summary(self, risk_data):
        """Display summary of risk metrics"""
        st.subheader('Risk Metrics Summary')
        
        metrics_df = pd.DataFrame({
            'Metric': ['Value at Risk (95%)', 'Max Drawdown', 'Current Volatility'],
            'Value': [
                f"{risk_data['var']:.2%}",
                f"{risk_data['drawdown'].min():.2%}",
                f"{risk_data['volatility'][-1]:.2%}"
            ]
        })
        
        st.table(metrics_df)
