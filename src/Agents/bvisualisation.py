import plotly.graph_objects as go
import streamlit as st


# Function to visualize portfolio impact
def visualize_portfolio_impact(prediction_result):
    st.write("### 📊 Portfolio Impact Visualization")

    # Extracting relevant data
    stock_symbol = prediction_result['stock_symbol']
    last_price = prediction_result['last_price']
    predicted_price = prediction_result['predicted_price']
    initial_market_value = prediction_result['initial_market_value']
    final_market_value = prediction_result['final_market_value']

    # Price Comparison Chart
    price_fig = go.Figure() #import plotly.graph_objects as go from this libary to get the requied graphs

    price_fig.add_trace(go.Bar(# sending the date and display in bar chart
        x=['Current', 'Predicted'],
        y=[last_price, predicted_price],
        name='Stock Price',
        marker_color='indigo'
    ))
    price_fig.update_layout(# diaplay names on x and y axis
        title=f"Price Comparison for {stock_symbol}",
        xaxis_title="Time",
        yaxis_title="Stock Price",
        barmode='group'
    )
    st.plotly_chart(price_fig)

    # Market Value Comparison
    value_fig = go.Figure()
    value_fig.add_trace(go.Pie(# sending the date and display in bar chart
        labels=['Initial Market Value', 'Predicted Market Value'],
        values=[initial_market_value, final_market_value],
        hole=.4
    ))
    value_fig.update_layout(#
        title=f"Market Value Change for {stock_symbol}"
    )
    st.plotly_chart(value_fig)

    # Percentage calulate Change Overview
    percent_change = ((float(predicted_price) - float(last_price)) / float(last_price)) * 100
    st.metric(label="📈 Predicted Percentage Change", value=f"{percent_change:.2f}%")

    # Risk Analysis Placeholder (Further analysis can be added)
    st.write("### 📉 Risk Exposure Analysis")
    st.info("Risk analysis module coming soon...")

