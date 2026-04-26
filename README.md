# Sales Trend Analysis and Forecasting Using Machine Learning

🚀 **Live App:** https://sales-trends-analysis-and-forecasting.streamlit.app/

A Machine Learning based Sales Forecasting Web Application built with Streamlit, Pandas, and Scikit-learn.

## Features
- **Dataset Upload**: Upload historical sales data (CSV) with `Date` and `Sales`.
- **Data Preprocessing**: Handles missing values, performs datetime conversions, sorting, and feature engineering (month, day, lags, rolling means).
- **Historical Data Visualization Dashboard**: Interactive charts of sales trends, moving averages, and monthly statistics using Plotly.
- **Machine Learning Model Training**: Train Linear Regression and Random Forest models with evaluation metrics (MAE, RMSE).
- **Sales Forecasting Module**: Predict future sales for user-specified days (e.g., next 30-90 days).

## Requirements
- Python 3.8+
- libraries listed in `requirements.txt`

## Installation
```bash
pip install -r requirements.txt
```

## Running the app
```bash
streamlit run app.py
```
