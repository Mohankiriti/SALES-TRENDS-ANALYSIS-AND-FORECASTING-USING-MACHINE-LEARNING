import pandas as pd
import numpy as np

import re

def detect_columns_and_currency(df):
    """
    Automatically detects the time column, the sales column, and any currency symbol.
    """
    time_col = None
    sales_col = None
    currency_symbol = ""
    qty_col = None
    
    currency_pattern = re.compile(r'([\$\€\£\¥\₹])')
    
    # First, detect quantity column to avoid confusing it with sales
    for col in df.columns:
        if col.lower() in ['quantity', 'qty', 'quantities', 'units', 'item count', 'count']:
            qty_col = col
            break
            
    # Then explicitly detect a proper sales column by keywords
    sales_keywords = ['price', 'sales', 'revenue', 'total', 'amount', 'value', 'cost']
    for col in df.columns:
        if col != qty_col and any(k in col.lower() for k in sales_keywords):
            sales_col = col
            break
            
    # Go through columns to find time, and fallback/currency extraction
    for col in df.columns:
        if time_col is None:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                time_col = col
            elif df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col].dropna().astype(str).head(5), errors='raise')
                    time_col = col
                except:
                    pass
                    
        # Check currency and numerics
        if col != time_col and col != qty_col:
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            temp_symbol = ""
            
            if df[col].dtype == 'object':
                sample = df[col].dropna().astype(str).head(20)
                for val in sample:
                    match = currency_pattern.search(val)
                    if match:
                        temp_symbol = match.group(1)
                        is_numeric = True
                        break
                
                if not is_numeric:
                    try:
                        clean_sample = sample.str.replace(r'[^\d.-]', '', regex=True)
                        clean_sample = clean_sample[clean_sample != '']
                        if len(clean_sample) > 0:
                            pd.to_numeric(clean_sample, errors='raise')
                            is_numeric = True
                    except:
                        pass
                        
            # If this is our sales col picked by name, record the symbol
            if col == sales_col and temp_symbol:
                currency_symbol = temp_symbol
                
            # If we still don't have a sales col, pick the first valid numeric one that isn't blacklisted
            if sales_col is None and is_numeric:
                col_lower = col.lower()
                if not any(ignore_term in col_lower for ignore_term in ['id', 'time', 'date', 'year', 'month', 'day', 'zip', 'code', 'discount']):
                    sales_col = col
                    currency_symbol = temp_symbol
                    
    if time_col is None:
        time_col = df.columns[0]
    if sales_col is None:
        # Fallback if no specific sales column found
        remaining_cols = [c for c in df.columns if c != time_col and c != qty_col]
        sales_col = remaining_cols[0] if remaining_cols else df.columns[0]
        
    return time_col, sales_col, currency_symbol, qty_col

def process_data(df):
    """
    Preprocess the uploaded sales data:
    - Automatically detect time, sales columns, and currency symbol
    - Handle missing values and convert currency to numeric
    - Convert time column to datetime format
    - Aggregate daily totals for duplicate dates
    - Sort chronologically
    - Generate derived columns: Year, Month, Day, 7-Day Average Sales, Average Sales per Month
    """
    data = df.copy()
    time_col, sales_col, currency_symbol, qty_col = detect_columns_and_currency(data)
    
    # 1. Handle missing Values
    data = data.dropna(subset=[time_col])
    
    if data[sales_col].dtype == 'object':
        data[sales_col] = data[sales_col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
        # Handle cases where replacing left empty strings
        data[sales_col] = data[sales_col].replace('', np.nan)
        data[sales_col] = pd.to_numeric(data[sales_col], errors='coerce')
        
    data[sales_col] = data[sales_col].ffill().bfill()
        
    # 2. Convert Time column to Datetime
    try:
        data['parsed_date'] = pd.to_datetime(data[time_col], format='%d-%m-%Y %H:%M', errors='coerce')
        data['parsed_date'] = data['parsed_date'].fillna(pd.to_datetime(data[time_col], dayfirst=True, errors='coerce'))
    except Exception:
        data['parsed_date'] = pd.to_datetime(data[time_col], dayfirst=True, errors='coerce')
        
    data = data.dropna(subset=['parsed_date'])
    
    data['Date_Only'] = data['parsed_date'].dt.date
    
    if qty_col:
        data[qty_col] = pd.to_numeric(data[qty_col], errors='coerce').fillna(1)
        
    # 3. Aggregate Daily Totals
    # Group by Date_Only and sum sales and quantity
    agg_dict = {sales_col: 'sum'}
    if qty_col:
        agg_dict[qty_col] = 'sum'
        
    counts = data.groupby('Date_Only').size().reset_index(name='trx_count')
    data = data.groupby('Date_Only', as_index=False).agg(agg_dict)
    
    # Prune deeply incomplete partial days (e.g. last day of collection) to prevent massive mathematical outlier drops
    data = data.merge(counts, on='Date_Only')
    median_trx = data['trx_count'].median()
    data = data[data['trx_count'] >= median_trx * 0.1]
    data = data.drop(columns=['trx_count'])
    
    # Standardize column names for output display
    # Provide Date as string explicitly so it never drops from rendering
    data['Date'] = pd.to_datetime(data['Date_Only']).dt.strftime('%Y-%m-%d')
    data['Parsed Date'] = pd.to_datetime(data['Date_Only'])
    data['parsed_date'] = data['Parsed Date']
    data['Total Sales'] = data[sales_col]
    if qty_col:
        data['Quantity'] = data[qty_col]
    
    # Retain standard features for models/charts
    data[time_col] = data['Date_Only']
    
    # 4. Sort Chronologically
    data = data.sort_values('parsed_date').reset_index(drop=True)
    
    # 5. Generate Derived Columns
    data['Year'] = data['parsed_date'].dt.year
    data['Month'] = data['parsed_date'].dt.month
    data['Day'] = data['parsed_date'].dt.day
    data['Week_Start'] = data['parsed_date'].dt.to_period('W').apply(lambda r: r.start_time)
    
    data['Standard_Time'] = data['parsed_date']
    data['Standard_Sales'] = data['Total Sales']
    
    data['7-Day Moving Average sales'] = data['Standard_Sales'].rolling(window=7, min_periods=1).mean()
    data['Monthly Average Sales'] = data.groupby(['Year', 'Month'])['Standard_Sales'].transform('mean')
    
    data['lag_1'] = data['Standard_Sales'].shift(1).ffill().bfill()
    data['lag_7'] = data['Standard_Sales'].shift(7).ffill().bfill()
    data['moving_avg_7_day'] = data['7-Day Moving Average sales']
    
    return data, time_col, sales_col, currency_symbol, qty_col
