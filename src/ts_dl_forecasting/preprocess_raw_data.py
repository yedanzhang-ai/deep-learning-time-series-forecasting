#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
preprocess_raw_data.py

This module contains the preprocessing pipeline for converting raw M5 data
(sales, prices, calendar) into clean, modeling-ready datasets.

It performs:
- Raw data loading
- Filtering by dept_id and store_id
- Reshaping sales (wide → long)
- Joining with calendar data
- Aggregating sales at the DEPT × STORE × DATE level
- Filtering time series with too many zero-sales days
- Creating price series at the same granularity
- Cleaning calendar events and categorical columns
- Saving processed outputs to CSV for downstream pipelines

This script can be run directly from CLI:

    python -m ts_forecasting.preprocess_raw_data \
        --dept_id HOUSEHOLD_2 \
        --store_id CA_2 \
        --output_dir data/processed
"""

import yaml
import pandas as pd
import numpy as np
from datetime import timedelta
import os
import argparse
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# Laod Raw Data
# ============================================================================

def load_raw_data_fn(data_info_dict):    
    
    """
    Load multiple raw CSV files into a dictionary of DataFrames.

    Args:
        data_info_dict (dict):
            Keys are dataset names (e.g., 'sales', 'price', 'calendar'),
            values are file paths.

    Returns:
        dict[str, pd.DataFrame]: A dictionary of loaded DataFrames.
    """
    res = {}    
    for df_name, data_file_path in data_info_dict.items():
        res[df_name] = pd.read_csv(data_file_path)        
    return res  

# ============================================================================
# Process Sales Data
# ============================================================================

def process_sales_data_fn(sales_df_raw, calendar_df_raw, dept_id, store_id):
    
    """
    Process sales data:
    - filter by dept_id & store_id
    - reshape from wide to long (d_1, d_2, … → rows)
    - merge with calendar to get actual dates
    - aggregate sales at DEPT × STORE × DATE level
    - filter out series with too many zero-sale days

    Returns:
        pd.DataFrame with columns:
            ['id', 'date', 'sales']
    """
    
    # Get the sales history of interested department and store
    sales = sales_df_raw[(sales_df_raw['dept_id']==dept_id) & (sales_df_raw['store_id']==store_id)]    
    sales_t = pd.melt(
        sales, 
        id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], 
        var_name='d', 
        value_name='sales')
    
    # Add calendar dates
    sales_cal = sales_t.merge(calendar_df_raw[['date', 'd']], on='d')
    sales_cal.drop(columns=['d'], inplace=True)

    # Aggregate sales at dept × store × date
    sales_dept_store=(
        sales_cal
        .groupby(['dept_id', 'store_id', 'date'])
        .agg(sales_agg=('sales', 'sum'))
        .reset_index()
    )

    # Format data
    
    sales_dept_store['id'] = sales_dept_store['dept_id'] + '_' + sales_dept_store['store_id']
    sales_dept_store.drop(columns=['dept_id', 'store_id'], inplace=True)    
    sales_dept_store = sales_dept_store[['id', 'date', 'sales_agg']]
    sales_dept_store.rename(columns={'sales_agg': 'sales'}, inplace=True)
    
    sales_dept_store['date'] =  pd.to_datetime(sales_dept_store['date'])
    sales_dept_store['sales'] = sales_dept_store['sales'].astype('float')
    sales_dept_store['id'] = sales_dept_store['id'].astype('category')
    
    return sales_dept_store

# ============================================================================
# Process Price Data
# ============================================================================

def process_price_data_fn(price_df_raw, sales_df_raw, calendar_df_raw, dept_id, store_id):
    
    """
    Process price data:
    - match items belonging to target DEPT × STORE
    - join with calendar to map week-based price to daily dates
    - aggregate mean price at DEPT × STORE × DATE
    """
    
    # Get the price data of interested department and store    
    item_dept_store = (
        sales_df_raw[
            (sales_df_raw['dept_id']==dept_id)&
            (sales_df_raw['store_id']==store_id)
        ][['item_id', 'dept_id', 'store_id']]
        .drop_duplicates()
    )
    
    price_dept_store = price_df_raw.merge(item_dept_store, on=['item_id', 'store_id'])
    
    # Get the calendar dates
    price_dept_store_cal = price_dept_store.merge(calendar_df_raw[['date', 'wm_yr_wk']].drop_duplicates(), on = ['wm_yr_wk'])
    price_dept_store_cal.drop(columns=['wm_yr_wk'], inplace=True)
    
    # Get the average prices of products at dept x store x date
    price_dept_store_cal = price_dept_store_cal.groupby(['dept_id', 'store_id', 'date'])['sell_price'].mean().reset_index()
    
    # Format data
    price_dept_store_cal['id'] = price_dept_store_cal['dept_id'] + '_' + price_dept_store_cal['store_id']
    price_dept_store_cal.drop(columns=['store_id', 'dept_id'], inplace=True)
    price_dept_store_cal = price_dept_store_cal[['id', 'date', 'sell_price']]
    
    price_dept_store_cal['date'] = pd.to_datetime(price_dept_store_cal['date'])
    price_dept_store_cal['sell_price'] = price_dept_store_cal['sell_price'].astype('float')
    price_dept_store_cal['id'] = price_dept_store_cal['id'].astype('category')
    
    return price_dept_store_cal

# ============================================================================
# Process Calendar Data
# ============================================================================

def process_calendar_data_fn(calendar_df_raw):
    
    """
    Clean calendar data:
    - drop columns not needed for modeling
    - convert event fields to categorical
    - convert certain numeric fields to categorical for embeddings
    """
    
    calendar_df = calendar_df_raw.drop(columns=['wm_yr_wk', 'weekday', 'd'])
    calendar_df['date'] = pd.to_datetime(calendar_df['date'])
    
    # Event fields to fill missing value and convert them to categorical data
    for col in ['event_name_1', 'event_name_2', 'event_type_1', 'event_type_2']:
        calendar_df[col] = calendar_df[col].fillna('0_NoEvent')
        calendar_df[col] = calendar_df[col].astype('category')
        
     # Convert numeric calendar features to categorical
    for col in ['wday', 'month']:
        calendar_df[col] = calendar_df[col].astype('category')
        
    return calendar_df  

# ============================================================================
# Orchestration Function
# ============================================================================

def orchestrate_process_raw_data_fn(data_info_dict, dept_id, store_id, output_dir):
    
    """
    Full raw data preprocessing pipeline:
    - loads raw sales/price/calendar
    - processes each component
    - saves cleaned results to output_dir

    Args:
        data_info_dict (dict): Mapping {name: file_path}.
        dept_id (str): M5 department ID.
        store_id (str): M5 store ID.
        output_dir (str): Directory to save processed CSVs.

    Returns:
        dict: Paths to processed data files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load raw inputs
    data = load_raw_data_fn(data_info_dict)
    
    # Apply processing
    sales_df = process_sales_data_fn(data['sales'], data['calendar'], dept_id, store_id)
    price_df = process_price_data_fn(data['price'], data['sales'], data['calendar'], dept_id, store_id)
    calendar_df = process_calendar_data_fn(data['calendar'])
    
    # Save outputs
    sales_path = os.path.join(output_dir, "sales_df.csv")
    price_path = os.path.join(output_dir, "price_df.csv")
    calendar_path = os.path.join(output_dir, "calendar_df.csv")
    
    sales_df.to_csv(sales_path, index=False)
    price_df.to_csv(price_path, index=False)
    calendar_df.to_csv(calendar_path, index=False)
    
    return {
        "sales": sales_path,
        "price": price_path,
        "calendar": calendar_path,
    }

# ============================================================================
# Main Entrypoint
# ============================================================================

if __name__ == "__main__":
    """
    Standalone script usage:
        python -m ts_forecasting.preprocess_raw_data \
            --dept_id HOUSEHOLD_2 \
            --store_id CA_2 \
            --output_dir data/processed

    This is useful for: 
    - preparing data for the deep-learning pipelines
    - reproducible preprocessing in MLOps workflows
    """
    
    parser = argparse.ArgumentParser(description="Preprocess raw M5 data.")
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

    data_info_dict = cfg["raw_data"]
    dept_id = cfg["preprocessing"]["dept_id"]
    store_id = cfg["preprocessing"]["store_id"]
    output_dir = cfg["preprocessing"]["output_dir"]

    # Ensure output_dir exists
    os.makedirs(output_dir, exist_ok=True)

    # Run preprocessing
    orchestrate_process_raw_data_fn(
        data_info_dict=data_info_dict,
        dept_id=dept_id,
        store_id=store_id,
        output_dir=output_dir
    )

    print("Preprocessing complete.")