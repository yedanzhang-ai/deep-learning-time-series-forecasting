#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
data.py

Shared data utilities for deep learning time series models (LSTM, TCN,
N-BEATS, TiDE, TFT, etc.) built with Darts.

Helper functions defined:
- Load the modeling-ready feature table from CSV.
- Identify which columns should be scaled vs. kept as raw covariates.
- Convert pandas DataFrames to Darts TimeSeries.
- Split TimeSeries into train/validation/test segments with horizon-aware logic.
- Scale time series using Darts' Scaler wrapper around sklearn scalers.

These helper functions are intended to be reused across multiple model
pipelines to ensure consistent preprocessing and data handling.
"""
import yaml
import warnings
import logging

import numpy as np
import pandas as pd


from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler

from sklearn.preprocessing import StandardScaler, RobustScaler

# ---------------------------------------------------------------------------
# Logging & warnings configuration
# ---------------------------------------------------------------------------

logging.getLogger().setLevel(logging.WARNING)
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Data loading & preprocessing utilities
# -------------------------------------------------------------------

def load_data_fn(data_file_path, id_var, target_var, dt_var, scales_vars):
    """
    Load raw data and identify non-scaled future covariates.

    Args:
        data_file_path (str): Path to input CSV file.
        id_var (str): Column name for time series identifier.
        target_var (str): Column name for the target variable to forecast.
        dt_var (str): Column name for the datetime variable.
        scales_vars (list[str]): List of feature names that will be scaled.

    Returns:
        tuple:
            df (pd.DataFrame): Loaded and sorted DataFrame.
        
        covariates_no_scale (list[str]): 
            Names of features treated as covariates that are not scaled.
    """
    
    logger.info("Loading data from %s", data_file_path)
    df = pd.read_csv(data_file_path)
    
    # Parse datetime column and sort dataframe
    df[dt_var] = pd.to_datetime(df[dt_var])
    df = df.sort_values(by=[id_var, dt_var])  
    
    # Identify covariates that are not scaled
    covariates_no_scale = [
        var 
        for var in df.columns.tolist() 
        if (var not in scales_vars) & (var not in [id_var, dt_var, target_var])
    ]

    return df, covariates_no_scale

def make_darts_ts_fn(df, dt_var, value_cols_lst, freq):
    """
    Convert a pandas DataFrame to a Darts TimeSeries object.

    Args:
        df (pd.DataFrame): Input DataFrame.
        dt_var (str): Name of the datetime column.
        value_cols_lst (list[str]): List of column names to be used as values.
        freq (str): Pandas/Darts-compatible frequency string.

    Returns:
        TimeSeries: Darts TimeSeries with float32 values and missing dates filled according to 
        the specified frequency
    """
    df_ts = TimeSeries.from_dataframe(
        df, 
        time_col = dt_var,
        value_cols = value_cols_lst,
        fill_missing_dates = True,
        freq = freq
    )
    return df_ts.astype(np.float32)

def split_data_fn(y, X_scale, X_no_scale, forecast_horizon):
    """ 
    Split data into train, validation, and test sets for time series.

    The split is horizon-aware:
    - ~85% for train, but capped to keep at least 15 * forecast_horizon points
      at the end of the series.
    - ~10% for validation, but ensures there are at least 10 * forecast_horizon
      points left for validation+test.
    - Remaining ~5% (or more if constrained by horizon) for test.

    Args:
        y (TimeSeries): Target time series.
        X_scale (TimeSeries): Covariates that will be scaled.
        X_no_scale (TimeSeries): Covariates that will not be scaled.
        forecast_horizon (int): Forecast horizon used to constrain split sizes and avoid leakage.

    Returns:
        tuple:
            train_end (int): Index where the training period ends.
            valid_end (int): Index where the validation period ends.
            y_train (TimeSeries): Training segment of target.
            y_valid (TimeSeries): Validation segment of target.
            y_test  (TimeSeries): Test segment of target.
            X_scale_train (TimeSeries): Training segment of scaled covariates.
            X_scale_valid (TimeSeries): Validation segment of scaled covariates.
            X_scale_test  (TimeSeries): Test segment of scaled covariates.
            X_no_scale_train (TimeSeries): Training segment of non-scaled covariates.
            X_no_scale_valid (TimeSeries): Validation segment of non-scaled covariates.
            X_no_scale_test  (TimeSeries): Test segment of non-scaled covariates.
    """
    n = len(y)
    
    # Ensure enough space at the tail for validation and test relative to horizon    
    train_end = int(min(n*0.85, n-forecast_horizon*15))
    valid_end = int(max(n*0.95, n-forecast_horizon*10))
    
    logger.info("Total length: %d | train_end: %d | valid_end: %d", n, train_end, valid_end)

    # Target split
    y_train = y[:train_end]
    y_valid = y[train_end : valid_end]
    y_test  = y[valid_end : ]
    
    # Scaled covariates splits
    X_scale_train = X_scale[:train_end]
    X_scale_valid = X_scale[train_end : valid_end]
    X_scale_test  = X_scale[valid_end : ]
    
    # Non-scaled covariates splits
    X_no_scale_train = X_no_scale[:train_end]
    X_no_scale_valid = X_no_scale[train_end : valid_end]
    X_no_scale_test  = X_no_scale[valid_end : ]

    return (
        train_end, 
        valid_end, 
        y_train,
        y_valid, 
        y_test, 
        X_scale_train, 
        X_scale_valid, 
        X_scale_test, 
        X_no_scale_train, 
        X_no_scale_valid, 
        X_no_scale_test
    )

def scale_data_fn(train, valid, test, model_name):
    """
    Fit a Scaler on the training set and apply to train/valid/test.

    This is aligned with MLOps best practices:
    - Fit transformations on training data only.
    - Apply the same transformation to validation/test.
    - Keep the fitted transformer for reproducibility and future inverse transform.

    Different scalers are used for models based on models' characteristics
        - LSTM / TiDE / TFT: StandardScaler (mean=0, std=1).
        - Others (TCN, N-BEATS, GRU): RobustScaler to reduce sensitivity to outliers.
    
    Args:
        train (TimeSeries): Training TimeSeries.
        valid (TimeSeries): Validation TimeSeries.
        test (TimeSeries): Test TimeSeries.

    Returns:
        tuple:
            var_scaler (Scaler): Fitted Darts Scaler.
            train_scaled (TimeSeries): Scaled training series.
            valid_scaled (TimeSeries): Scaled validation series.
            test_scaled (TimeSeries): Scaled test series.
            all_scaled (TimeSeries): Concatenation of all scaled splits.
    """
    
    if model_name.lower() in ['lstm', 'tide', 'tft']:
        base_scaler = StandardScaler()   
        logger.info("Using StandardScaler for model %s", model_name)
    else:
        base_scaler = RobustScaler()  
        logger.info("Using RobustScaler for model %s", model_name)
    
    var_scaler = Scaler(scaler=base_scaler)  
    
     # Fit only on the training period
    var_scaler.fit(train)
    train_scaled = var_scaler.transform(train)
    valid_scaled = var_scaler.transform(valid)
    test_scaled  = var_scaler.transform(test)
    
    # Concatenate for future use 
    all_scaled   = concatenate([train_scaled, valid_scaled, test_scaled], axis=0)

    return (var_scaler, train_scaled, valid_scaled, test_scaled, all_scaled)

def build_dataset_from_config(config_path, model_name):
    """
    Build train/valid/test datasets and scalers from a YAML config.

    Args:
        config_path (str): Path to the YAML file containing the `data` block.
        model_name (str): Name of the model type (e.g., 'lstm', 'tide', 'tft', 'tcn', 'nbeats'),
            used by `scale_data_fn` to pick StandardScaler vs RobustScaler.

    Returns:
        dict[str, Any]:
            A dictionary with keys:
               
                - "y"                      : full target TimeSeries
                - "X_scale"                : full scaled-candidate covariates TimeSeries
                - "X_no_scale"             : full non-scaled covariates TimeSeries

                - "y_train", "y_valid", "y_test"
                - "X_scale_train", "X_scale_valid", "X_scale_test"
                - "X_no_scale_train", "X_no_scale_valid", "X_no_scale_test"

                - "y_scaler"               : fitted Scaler for target
                - "y_train_scaled", "y_valid_scaled", "y_test_scaled", "y_all_scaled"
                - "X_scaler"               : fitted Scaler for scaled covariates
                - "X_train_scaled", "X_valid_scaled", "X_test_scaled", "X_all_scaled"

                - "freq"                   : frequency string from config
                - "forecast_horizon"       : forecast horizon from config
    """
    # ------------------------------------------------------------------
    # 1. Load YAML configuration
    # ------------------------------------------------------------------
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]

    data_file_path = data_cfg["file_path"]
    id_var = data_cfg["id_var"]
    dt_var = data_cfg["dt_var"]
    target_var = data_cfg["target_var"]
    freq = data_cfg["freq"]
    forecast_horizon = data_cfg["forecast_horizon"]
    scale_vars = data_cfg["scale_vars"]

    # ------------------------------------------------------------------
    # 2. Load DataFrame + identify non-scaled covariates
    # ------------------------------------------------------------------
    df, covariates_no_scale = load_data_fn(
        data_file_path=data_file_path,
        id_var=id_var,
        target_var=target_var,
        dt_var=dt_var,
        scales_vars=scale_vars,
    )

    # ------------------------------------------------------------------
    # 3. Build Darts TimeSeries for y, X_scale, X_no_scale
    # ------------------------------------------------------------------
    y = make_darts_ts_fn(df, dt_var, [target_var], freq)
    X_scale = make_darts_ts_fn(df, dt_var, scale_vars, freq)
    X_no_scale = make_darts_ts_fn(df, dt_var, covariates_no_scale, freq)

    # ------------------------------------------------------------------
    # 4. Split into train / valid / test
    # ------------------------------------------------------------------
    (
        train_end,
        valid_end,
        y_train,
        y_valid,
        y_test,
        X_scale_train,
        X_scale_valid,
        X_scale_test,
        X_no_scale_train,
        X_no_scale_valid,
        X_no_scale_test,
    ) = split_data_fn(y, X_scale, X_no_scale, forecast_horizon)

    # ------------------------------------------------------------------
    # 5. Scale target and scaled covariates
    # ------------------------------------------------------------------
    (y_scaler, y_train_scaled, y_valid_scaled, y_test_scaled, y_all_scaled) = scale_data_fn(
        train=y_train,
        valid=y_valid,
        test=y_test,
        model_name=model_name,
    )

    (X_scaler, X_train_scaled, X_valid_scaled, X_test_scaled, X_all_scaled) = scale_data_fn(
        train=X_scale_train,
        valid=X_scale_valid,
        test=X_scale_test,
        model_name=model_name,
    )

    # ------------------------------------------------------------------
    # 6. Package everything into a dictionary
    # ------------------------------------------------------------------
    return {        
        # Full TimeSeries
        "y": y,
        "X_scale": X_scale,
        "X_no_scale": X_no_scale,

        # Data splits
        "y_train": y_train,
        "y_valid": y_valid,
        "y_test": y_test,
        "X_scale_train": X_scale_train,
        "X_scale_valid": X_scale_valid,
        "X_scale_test": X_scale_test,
        "X_no_scale_train": X_no_scale_train,
        "X_no_scale_valid": X_no_scale_valid,
        "X_no_scale_test": X_no_scale_test,

        # Target scaling
        "y_scaler": y_scaler,
        "y_train_scaled": y_train_scaled,
        "y_valid_scaled": y_valid_scaled,
        "y_test_scaled": y_test_scaled,
        "y_all_scaled": y_all_scaled,

        # Covariate scaling
        "X_scaler": X_scaler,
        "X_train_scaled": X_train_scaled,
        "X_valid_scaled": X_valid_scaled,
        "X_test_scaled": X_test_scaled,
        "X_all_scaled": X_all_scaled,

        # Metadata
        "freq": freq,
        "forecast_horizon": forecast_horizon,
        "id_var": id_var,
        "dt_var": dt_var,
        "target_var": target_var,
        "scale_vars": scale_vars,
    }

