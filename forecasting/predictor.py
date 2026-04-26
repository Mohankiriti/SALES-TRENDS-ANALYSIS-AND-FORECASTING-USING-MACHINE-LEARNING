import pandas as pd
import numpy as np
from datetime import timedelta

def forecast_future(df, model, days=365):
    """
    Generate sales predictions for the next `days` step-by-step iteratively.
    Can forecast up to 365 days.
    """
    last_date = df['Standard_Time'].max()
    
    # Generate future daily dates
    future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
    future_df = pd.DataFrame({'Standard_Time': future_dates})
    
    # Basic features without Year to prevent uncontrolled linear extrapolation
    future_df['Month'] = future_df['Standard_Time'].dt.month
    future_df['Day'] = future_df['Standard_Time'].dt.day
    future_df['DayOfWeek'] = future_df['Standard_Time'].dt.dayofweek
    
    predictions = []
    
    # Get last known sales points to kickstart moving averages and lags
    recent_history = df['Standard_Sales'].tail(14).tolist()
    
    # Historical baselines to prevent downward spirals
    historical_mean = df['Standard_Sales'].mean()
    historical_min = max(0, df['Standard_Sales'].quantile(0.05) * 0.8) # 5th percentile floor
    historical_max = df['Standard_Sales'].quantile(0.95) * 1.5
    
    for i in range(days):
        row = future_df.iloc[i:i+1].copy()
        
        # Calculate lags based on current recent history stream
        lag_1 = recent_history[-1]
        lag_7 = recent_history[-7]
        moving_avg_7_day = np.mean(recent_history[-7:])
        
        row['lag_1'] = lag_1
        row['lag_7'] = lag_7
        row['moving_avg_7_day'] = moving_avg_7_day
        
        features = ['Month', 'Day', 'DayOfWeek', 'lag_1', 'lag_7', 'moving_avg_7_day']
        
        # Predict
        pred = model.predict(row[features])[0]
        
        # Apply mean reversion to prevent autoregressive decay (downward spiral)
        # This safely pulls the prediction towards the overall average range
        pred = (pred * 0.65) + (historical_mean * 0.35)
        
        # Enforce realistic bounding limits
        if pred < historical_min:
            pred = historical_min
        elif pred > historical_max:
            pred = historical_max
            
        predictions.append(pred)
        
        # Insert prediction to history for next rows computation
        recent_history.append(pred)
        # Keep list size bounded
        recent_history.pop(0)
        
    future_df['Predicted_Sales'] = predictions
    
    # Cleanup display dataframe
    display_df = future_df[['Standard_Time', 'Predicted_Sales']].rename(columns={'Standard_Time': 'Date'})
    # Round predictions to 2 decimals for cleaner view
    display_df['Predicted_Sales'] = display_df['Predicted_Sales'].round(2)
    
    return display_df
