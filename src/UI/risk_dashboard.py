import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def plot_var_chart(var_value, returns):
    """Create VaR visualization"""
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=returns, nbinsx=50, name='Returns Distribution'))
    fig.add_vline(x=var_value, line_color='red', line_dash='dash')
    fig.update_layout(title='Value at Risk Distribution',
                     xaxis_title='Returns',
                     yaxis_title='Frequency')
    return fig

def plot_drawdown_chart(drawdown_series):
    """Create drawdown visualization"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=drawdown_series.index, 
                            y=drawdown_series.values,
                            mode='lines',
                            name='Drawdown'))
    fig.update_layout(title='Portfolio Drawdown',
                     xaxis_title='Date',
                     yaxis_title='Drawdown %')
    return fig

def plot_correlation_heatmap(correlation_matrix):
    """Create correlation heatmap"""
    fig = px.imshow(correlation_matrix,
                    labels=dict(x='Asset', y='Asset', color='Correlation'),
                    color_continuous_scale='RdBu')
    fig.update_layout(title='Asset Correlation Heatmap')
    return fig

def display_risk_metrics(risk_data):
    """Display risk metrics dashboard"""
    st.subheader('Risk Metrics Dashboard')
    
    # VaR Chart
    st.plotly_chart(plot_var_chart(risk_data['var'], risk_data['returns']))
    
    # Drawdown Chart
    st.plotly_chart(plot_drawdown_chart(risk_data['drawdown']))
    
    # Correlation Heatmap
    if 'correlation' in risk_data:
        st.plotly_chart(plot_correlation_heatmap(risk_data['correlation']))
    
    # Risk Metrics Summary
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
