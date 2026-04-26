import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def render_dashboard(df, time_col, sales_col, currency_symbol=""):
    """
    Render historical sales data visualization dashboard.
    - Statistics summary table
    - Sales trend line chart with anomalies highlighted
    - Monthly sales analysis (bar graph)
    - Weekly trend analysis (7-day moving average)
    - Bar chart for average sales per 7-day period
    """
    curr = f" ({currency_symbol})" if currency_symbol else ""
    
    # 0. Statistics Summary Table
    st.write("### Dataset Statistics Summary")
    st.markdown(f"Detailed numeric overview of `Total Sales` volume showing the data spread, mean, and limits.")
    
    # Format the describe table to display only the numeric values without currency symbols
    # Explicitly calculate from the correctly aggregated daily sales column
    desc_df = df[['Total Sales']].describe().T
    st.dataframe(desc_df, use_container_width=True)

    # Calculate Anomalies (defined as points 2 standard deviations away from the mean)
    mean_sales = df['Standard_Sales'].mean()
    std_sales = df['Standard_Sales'].std()
    
    df_anom = df.copy()
    df_anom['Anomaly'] = np.where(
        (df_anom['Standard_Sales'] > mean_sales + 2 * std_sales) | 
        (df_anom['Standard_Sales'] < mean_sales - 2 * std_sales), 
        True, False
    )

    # 1. Sales Trend Line Chart (with Anomalies)
    st.write("---")
    st.write("### Sales Trend Line Chart")
    st.markdown("Displays the day-to-day fluctuations in sales over time. **Special events or anomalies** (outliers > 2 standard deviations from the mean) are highlighted in <span style='color:red'>**red markers**</span> to help identify abnormal spikes or drops.", unsafe_allow_html=True)
    fig_trend = go.Figure()
    
    # Base Line
    fig_trend.add_trace(go.Scatter(
        x=df_anom['Standard_Time'], 
        y=df_anom['Standard_Sales'],
        mode='lines',
        name='Sales Trend',
        line=dict(color='#1f77b4')
    ))
    
    # Anomalies Scatter
    anomalies = df_anom[df_anom['Anomaly']]
    if not anomalies.empty:
        fig_trend.add_trace(go.Scatter(
            x=anomalies['Standard_Time'], 
            y=anomalies['Standard_Sales'],
            mode='markers',
            marker=dict(color='red', size=8, line=dict(color='DarkSlateGrey', width=1)),
            name='Anomaly/Special Event'
        ))
        
    fig_trend.update_layout(
        title=f'Continuous Sales Trend over Time{curr}', 
        xaxis_title="Date", 
        yaxis_title=f"Sales Quantity / Value{curr}",
        hovermode="x unified"
    )
    if currency_symbol:
        fig_trend.update_yaxes(tickprefix=currency_symbol)
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # 2. Monthly Sales Analysis (Bar Graph)
    st.write("---")
    st.write("### Monthly Sales Analysis")
    st.markdown("Aggregates the total sales volume for each month. This helps in understanding wider seasonal patterns and comparing month-over-month performance.")
    monthly_sales = df.groupby(['Year', 'Month'])['Standard_Sales'].sum().reset_index()
    monthly_sales['Year_Month'] = monthly_sales['Year'].astype(str) + '-' + monthly_sales['Month'].astype(str).str.zfill(2)
    fig_monthly = px.bar(monthly_sales, x='Year_Month', y='Standard_Sales', 
                         title=f'Total Sales per Month{curr}',
                         labels={'Year_Month': 'Month', 'Standard_Sales': f'Total Sales{curr}'})
    if currency_symbol:
        fig_monthly.update_yaxes(tickprefix=currency_symbol)
    st.plotly_chart(fig_monthly, use_container_width=True)
    
    # 3. Weekly Trend Analysis - 7-Day Moving Average
    st.write("---")
    st.write("### Weekly Trend Analysis (7-Day Moving Average)")
    st.markdown("Overlays the raw daily sales data with a smoothed **7-Day Moving Average**. This statistical smoothing removes daily volatility (like weekend drops) and makes the true weekly trajectory easily readable.")
    
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=df['Standard_Time'], y=df['Standard_Sales'], 
                                mode='lines', name='Daily Sales', opacity=0.4))
    fig_ma.add_trace(go.Scatter(x=df['Standard_Time'], y=df['moving_avg_7_day'], 
                                mode='lines', name='7-Day Moving Avg', line=dict(color='red', width=3)))
    fig_ma.update_layout(
        title=f"Daily Sales vs 7-Day Moving Average{curr}", 
        xaxis_title="Date", 
        yaxis_title=f"Sales{curr}",
        hovermode="x unified"
    )
    if currency_symbol:
        fig_ma.update_yaxes(tickprefix=currency_symbol)
    st.plotly_chart(fig_ma, use_container_width=True)
    
    # 4. Bar Chart Showing Average Sales for each 7-Day period
    st.write("---")
    st.write("### Average Sales per 7-Day Period (Weekly Performance)")
    st.markdown("Calculates the average metric over grouped 7-day intervals starting from the first recorded date. Useful to judge base weekly performance uncoupled from calendar months.")
    # Group the dataframe into weekly intervals starting early on
    df['Week_Start_Date'] = df['Standard_Time'].dt.to_period('W').apply(lambda r: r.start_time)
    weekly_avg = df.groupby('Week_Start_Date')['Standard_Sales'].mean().reset_index()
    
    fig_weekly = px.bar(weekly_avg, x='Week_Start_Date', y='Standard_Sales', 
                        title=f'Average Sales Grouped by 7-Day Period{curr}',
                        labels={'Week_Start_Date': 'Week Starting', 'Standard_Sales': f'Average Sales Volume{curr}'})
    if currency_symbol:
        fig_weekly.update_yaxes(tickprefix=currency_symbol)
    st.plotly_chart(fig_weekly, use_container_width=True)

