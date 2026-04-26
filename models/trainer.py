import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

def train_evaluate_model(df):
    """
    Train and evaluate multiple machine learning regression models.
    Including: Linear Regression, Decision Tree, Random Forest, XGBoost and Logistic Regression.
    """
    # Features explicitly mentioned as time-based
    # We drop 'Year' because it causes negative extrapolation in future predictions for linear models.
    # We add DayOfWeek directly here for structural daily seasonality mapping.
    df['DayOfWeek'] = df['Standard_Time'].dt.dayofweek
    features = ['Month', 'Day', 'DayOfWeek', 'lag_1', 'lag_7', 'moving_avg_7_day']
    X = df[features]
    y = df['Standard_Sales']
    
    # Split chronologically - strict 10 day holdout extraction per user request
    split_idx = len(df) - 10 if len(df) > 10 else max(1, len(df) - 1)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost Regressor": XGBRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    metrics = {}
    fitted_models = {}
    test_results = {}
    
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
                
            preds = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            
            metrics[name] = {"MAE": mae, "RMSE": rmse, "R² Score": r2}
            fitted_models[name] = model
            
            # Generate test extraction for validation visualizer
            res_df = pd.DataFrame({
                'Date': df.iloc[split_idx:]['Standard_Time'].dt.strftime('%Y-%m-%d'),
                'Actual Sales': y_test.values,
                'Predicted Sales': preds,
                'Difference (Predicted - Actual)': preds - y_test.values
            })
            test_results[name] = res_df
            
        except Exception as e:
            # Fallback if model fails
            metrics[name] = {"MAE": np.nan, "RMSE": np.nan, "R² Score": np.nan}
            print(f"Failed to train {name}: {e}")
            
    return fitted_models, metrics, test_results
