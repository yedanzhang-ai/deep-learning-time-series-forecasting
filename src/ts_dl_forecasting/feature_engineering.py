#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
feature_engineering.py

This module performs feature engineering for time series forecasting.

High-level steps:
1. Load preprocessed data (sales, prices, calendar) from CSV paths.
2. Convert to time series indexed by date and resampled to a given frequency.
3. Build:
   - calendar-based features (day, month, year, cyclical encodings),
   - event-based features (one-hot encoded event types, scaled year, event flags).
4. Join sales, price, and calendar/event features into a single feature table.
5. Save the engineered feature table to disk for downstream modeling.

The module can be executed as a script:

    python -m ts_dl_forecasting.feature_engineering --config config/feature_eng.yaml

where the YAML config specifies:
- preprocessed_data: paths to sales_df, price_df, calendar_df
- feature_engineering: dt_var, target_var, freq, output_dir
"""

import os
import logging
import warnings
import argparse
from typing import Dict

import yaml
import ast
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# ---------------------------------------------------------------------------
# Logging & warnings configuration
# ---------------------------------------------------------------------------

logging.getLogger().setLevel(logging.WARNING)
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_preprocessed_data_fn(data_info_dict):    
    """
    Load preprocessed CSV files into pandas DataFrames.

    Args:
        data_info_dict (dict):
            Mapping from dataset name to CSV path, e.g.:
            {
                "sales_df": "data/processed/sales_df.csv",
                "price_df": "data/processed/price_df.csv",
                "calendar_df": "data/processed/calendar_df.csv",
            }

    Returns:
        dict[str, pd.DataFrame]:
            Dictionary of loaded DataFrames keyed by the same names.
    """
    res = {}
    
    for df_name, data_file_path in data_info_dict.items():
        logger.info(f"Loading {df_name} from {data_file_path}")
        res[df_name] = pd.read_csv(data_file_path)
        
    return res  


# ---------------------------------------------------------------------------
# Make time series data
# ---------------------------------------------------------------------------

def get_ts_fn(df, dt_var, freq):
    """
    Convert a DataFrame to a time-indexed DataFrame and resample to a given frequency.

    Args:
        df (pd.DataFrame): Input DataFrame containing a datetime column.
        dt_var (str): Name of the datetime column.
        freq (str): Resampling frequency string.

    Returns:
        pd.DataFrame: DataFrame indexed by datetime at the specified frequency.
    """
    res = df.copy()
    res[dt_var] = pd.to_datetime(res[dt_var])
    res = res.set_index(dt_var).resample(freq).sum()
    return res

# ---------------------------------------------------------------------------
# Generate calendar-based features
# ---------------------------------------------------------------------------

def get_calendar_feats_fn(ts):
    """
    Create calendar/time-based features from a time-indexed DataFrame.
    Uses the index as the date axis to derive:
        - quarter, dayofmonth, dayofyear
        - cyclical encodings (sin/cos) for these temporal components.

    Args:
        ts (pd.DataFrame): Time-indexed DataFrame including 'wday' and 'month'.

    Returns:
        pd.DataFrame: Calendar feature DataFrame (numeric features only).
    """
    
    calendar_feats = ts[['wday', 'month']]
        
    calendar_feats['quarter'] = ts.index.quarter
    calendar_feats['dayofmonth'] = ts.index.dayofweek 
    calendar_feats['dayofyear'] = ts.index.dayofyear 
    

    calendar_feats['quarter_sin'] = np.sin(calendar_feats["quarter"] / calendar_feats["quarter"].max() * 2 * np.pi)
    calendar_feats['quarter_cos'] = np.cos(calendar_feats["quarter"] / calendar_feats["quarter"].max() * 2 * np.pi)
    calendar_feats['month_sin'] = np.sin(calendar_feats["month"] / calendar_feats["month"].max() * 2 * np.pi)
    calendar_feats['month_cos'] = np.cos(calendar_feats["month"] / calendar_feats["month"].max() * 2 * np.pi)
    calendar_feats['dayofmonth_sin'] = np.sin(calendar_feats["dayofmonth"] / calendar_feats["dayofmonth"].max() * 2 * np.pi)
    calendar_feats['dayofmonth_cos'] = np.cos(calendar_feats["dayofmonth"] / calendar_feats["dayofmonth"].max() * 2 * np.pi)
    calendar_feats['dayofyear_sin'] = np.sin(calendar_feats["dayofyear"] / calendar_feats["dayofyear"].max() * 2 * np.pi)
    calendar_feats['dayofyear_cos'] = np.cos(calendar_feats["dayofyear"] / calendar_feats["dayofyear"].max() * 2 * np.pi)
    calendar_feats['wday_sin'] = np.sin(calendar_feats["wday"] / calendar_feats["wday"].max() * 2 * np.pi)
    calendar_feats['wday_cos'] = np.cos(calendar_feats["wday"] / calendar_feats["wday"].max() * 2 * np.pi)

    calendar_feats_df = calendar_feats.drop(
        columns=['wday', 'month', 'quarter', 'dayofmonth', 'dayofyear']
    )

    return calendar_feats_df    

# ---------------------------------------------------------------------------
# Generate event-based features
# ---------------------------------------------------------------------------

def get_event_feats_fn(ts):
    """
    Create event-based features from a time-indexed calendar DataFrame.

    Steps:
        1. Convert event_name_* into binary flags ("0_NoEvent" → 0, else 1).
        2. Scale 'year' using StandardScaler.
        3. One-hot encode event_type_* using OneHotEncoder (drop='first').
        4. Concatenate all into a single numeric feature matrix.

    Args:
        ts (pd.DataFrame): Calendar DataFrame indexed by date.

    Returns:
        pd.DataFrame: Event feature DataFrame with numeric columns.
    """
    event_feats = ts[['year', 'event_name_1', 'event_name_2', 'event_type_1', 'event_type_2']]
    event_feats['event_name_1'] = np.where(ts['event_name_1']=='0_NoEvent', 0, 1)
    event_feats['event_name_2'] = np.where(ts['event_name_2']=='0_NoEvent', 0, 1)

    event_ohe = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
    year_scaler = StandardScaler()

    transformers = [('num', year_scaler, ['year']),
                    ('cat', event_ohe, ['event_type_1', 'event_type_2']),
                    ('num_pass', 'passthrough', ['event_name_1', 'event_name_2'])]

    col_trans = ColumnTransformer(transformers = transformers)
    event_feats_trans = col_trans.fit_transform(event_feats)
    event_cat_names = col_trans.named_transformers_['cat'].get_feature_names_out(['event_type_1', 'event_type_2']).tolist()
    event_feats_trans_cols = ['year'] + event_cat_names + ['event_name_1', 'event_name_2']
    event_feats_trans_df = pd.DataFrame(event_feats_trans,
                                        index=event_feats.index,
                                        columns=event_feats_trans_cols)
    
    return event_feats_trans_df

# ---------------------------------------------------------------------------
# Orchestration function
# ---------------------------------------------------------------------------

def orchestrate_feature_engineering_fn(
    data_info_dict,
    dt_var,   
    freq,    
    output_dir
):
    """
    Orchestrate the full feature engineering pipeline.

    Steps:
        1. Load preprocessed sales, price, and calendar data.
        2. Convert calendar to time series and compute calendar & event features.
        3. Merge sales and price, resample to time series.
        4. Concatenate all features by date index.
        5. Save final feature table as CSV.

    Args:
        data_info_dict (dict):
            Mapping { "sales_df": path, "price_df": path, "calendar_df": path }.
        dt_var (str): Name of the datetime column in input DataFrames.       
        freq (str): Time series frequency.      
        output_dir (str): Directory to save engineered features.

    Returns:
        dict[str, str]: Mapping of output dataset names to file paths.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    data = load_preprocessed_data_fn(data_info_dict)

    sales_df = data['sales_df']
    price_df = data['price_df']
    calendar_df = data['calendar_df']

    calendar_ts = get_ts_fn(calendar_df, dt_var, freq)
    calendar_feats_df = get_calendar_feats_fn(calendar_ts)
    event_feats_trans_df = get_event_feats_fn(calendar_ts)

    sales_price_df=sales_df.merge(price_df, on=['id', 'date'])
    sales_price_ts = get_ts_fn(sales_price_df, dt_var, freq)

    sales_forecast_features = pd.concat(
        [sales_price_ts, calendar_feats_df, event_feats_trans_df], 
        axis=1, 
        join='inner'
    )
    
    sales_forecast_features = sales_forecast_features.reset_index()
    
    sales_forecast_features_path = os.path.join(output_dir, "sales_forecast_features.csv")
    
    sales_forecast_features.to_csv(sales_forecast_features_path, index=False)
    
    logger.info(f"Saved engineered features to {sales_forecast_features_path}")
    
    return {"sales_forecast_features": sales_forecast_features_path}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Engineer Featuers for Forecasting")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file."
    )

    args = parser.parse_args()
    
    # Load YAML configuration
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    data_info_dict = cfg["preprocessed_data"]
    dt_var = cfg["feature_engineering"]['dt_var']    
    freq = cfg["feature_engineering"]['freq']    
    output_dir = cfg["feature_engineering"]['output_dir']
    
    # Ensure output_dir exists
    os.makedirs(output_dir, exist_ok=True)

    # Run preprocessing
    orchestrate_feature_engineering_fn(
        data_info_dict=data_info_dict,
        dt_var=dt_var,       
        freq=freq,       
        output_dir=output_dir
    )

    print("Feature Engineering complete.")
    
    
    

