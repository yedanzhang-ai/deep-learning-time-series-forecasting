#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
pipeline.py

End-to-end pipeline for:

1. Building dataset from config (data.yaml).
2. Hyperparameter tuning with Optuna (using model_X.yaml).
3. Retraining the best model on train+valid.
4. Forecasting on test set.
5. Computing and saving evaluation metrics + best hyperparameters.

This works for any supported model:
- gru, lstm, nbeats, tcn, tide, tft

Usage (from repo root, with .venv activated):

    python -m ts_dl_forecasting.pipeline --config config/experiment_lstm.yaml
"""

import os
import logging
import warnings
import argparse
from typing import Dict, Any

import yaml
import torch
import pandas as pd
import matplotlib.pyplot as plt

from darts import concatenate
from darts.metrics import rmse, mae, mape, r2_score
from pytorch_lightning.callbacks import EarlyStopping

from ts_dl_forecasting.data import build_dataset_from_config
from ts_dl_forecasting.models import build_model
from ts_dl_forecasting.tuning import tune_model_fn, load_model_config

# ---------------------------------------------------------------------------
# Logging / warnings configuration
# ---------------------------------------------------------------------------

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)

logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.ERROR)
logging.getLogger("darts").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GPU / CPU usage depending on GPU availability
# ---------------------------------------------------------------------------

USE_GPU = torch.cuda.is_available()

# ---------------------------------------------------------------------------
# Covariate usage (must stay in sync with tuning.py)
# ---------------------------------------------------------------------------

FUTURE_COV_MODELS = {"gru", "lstm", "tide", "tft"}
PAST_COV_MODELS = {"nbeats", "tcn"}


# ---------------------------------------------------------------------------
# Core experiment runner
# ---------------------------------------------------------------------------

def run_experiment(experiment_config_path):
    """
    Run a full training + tuning + evaluation pipeline for a single model.

    The experiment config YAML should look like:

        experiment:
          model_name: "lstm"
          data_config: "config/data.yaml"
          model_config: "config/model_lstm.yaml"
          output_dir: "outputs/lstm"

    Steps:
    - Load dataset using data_config.
    - Tune model hyperparameters using model_config.
    - Retrain the best model on train+valid.
    - Forecast on test set and compute metrics.
    - Save metrics + best params under output_dir.

    Args:
        experiment_config_path (str): Path to experiment YAML.

    Returns:
        dict: Dictionary with metrics and paths to saved artifacts.
    """
    # ----------------------------------------------------------
    # 1) Load experiment-level config
    # ----------------------------------------------------------
    with open(experiment_config_path, "r") as f:
        full_cfg = yaml.safe_load(f)

    exp_cfg = full_cfg["experiment"]
    model_name = exp_cfg["model_name"].lower()
    data_config_path = exp_cfg.get("data_config", "config/data.yaml")
    model_config_path = exp_cfg.get("model_config", f"config/model_{model_name}.yaml")
    output_dir = exp_cfg.get("output_dir", f"outputs/{model_name}")

    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Running experiment for model '{model_name}'")
    logger.info(f"  data_config:  {data_config_path}")
    logger.info(f"  model_config: {model_config_path}")
    logger.info(f"  output_dir:   {output_dir}")

    # Load model-specific config (to get training params, etc.)
    model_cfg = load_model_config(model_config_path)

    n_epochs_train = model_cfg.get("n_epochs_train", 80)
    early_stop_patience = model_cfg.get("early_stop_patience", 10)
    seed = model_cfg.get("seed", 3)   

    # EarlyStopping for final training
    early_stop_train = EarlyStopping(
        monitor="train_loss",
        mode="min",
        patience=early_stop_patience,
        min_delta=1e-4,
        verbose=False,
    )

    # ----------------------------------------------------------
    # 2) Build dataset
    # ----------------------------------------------------------
    dataset = build_dataset_from_config(config_path=data_config_path, model_name=model_name)
    
    y_train_scaled = dataset["y_train_scaled"]
    y_valid_scaled = dataset["y_valid_scaled"]
    y_test = dataset["y_test"]
    y_scaler = dataset["y_scaler"]

    X_train_scaled = dataset["X_train_scaled"]
    X_valid_scaled = dataset["X_valid_scaled"]
    X_test_scaled = dataset["X_test_scaled"]
    X_no_scale_train = dataset["X_no_scale_train"]
    X_no_scale_valid = dataset["X_no_scale_valid"]
    X_no_scale_test = dataset["X_no_scale_test"]

    forecast_horizon = dataset["forecast_horizon"]

    # ----------------------------------------------------------
    # 3) Combine covariates and targets for tuning & retraining
    # ----------------------------------------------------------

    # Time-wise concat for target: train + valid   
    y_train_valid_scaled = concatenate([y_train_scaled, y_valid_scaled])

    # Feature-wise concatenation for covariates:
    #   scaled (e.g. price) + non-scaled (e.g. calendar/event features)
    X_train_all = concatenate([X_no_scale_train, X_train_scaled], axis=1)
    X_valid_all = concatenate([X_no_scale_valid, X_valid_scaled], axis=1)
    X_test_all = concatenate([X_no_scale_test, X_test_scaled], axis=1)

    # Time-wise concatenation for covariates (train+valid, all)       
    X_train_valid_all = concatenate([X_train_all, X_valid_all])
    X_all = concatenate([X_train_valid_all, X_test_all])

    # y_valid (unscaled) needed for evaluation in tuning
    y_valid = dataset["y_valid"]

    # ----------------------------------------------------------
    # 4) Hyperparameter tuning (Optuna)
    # ----------------------------------------------------------
    logger.info("Starting hyperparameter tuning with Optuna ...")

    study = tune_model_fn(
        model_name=model_name,
        y_train_scaled=y_train_scaled,
        X_train_all=X_train_all,
        y_valid=y_valid,
        X_train_valid_all=X_train_valid_all,
        y_scaler=y_scaler,
        forecast_horizon=forecast_horizon,
        use_gpu = USE_GPU,
        config_path=model_config_path,
    )

    best_params = study.best_params
    logger.info(f"Best trial value (MAE): {study.best_value:.4f}")
    logger.info(f"Best hyperparameters: {best_params}")

    # Save best params to YAML for reproducibility
    best_params_path = os.path.join(output_dir, "best_params.yaml")
    with open(best_params_path, "w") as f:
        yaml.safe_dump({"best_params": best_params}, f)

    # ----------------------------------------------------------
    # 5) Retrain the tuned model on train+valid
    # ----------------------------------------------------------
    logger.info("Retraining final model on train+valid with best hyperparameters ...")

    final_model = build_model(
        model_name=model_name,
        hp=best_params,
        n_epochs=n_epochs_train,
        early_stop=early_stop_train,
        forecast_horizon=forecast_horizon,
        SEED=seed,
        use_gpu=USE_GPU,
    )

    # Fit with correct covariate type
    if model_name in FUTURE_COV_MODELS:
        final_model.fit(
            series=y_train_valid_scaled,
            future_covariates=X_train_valid_all,
            verbose=False,
        )
        forecast_test_scaled = final_model.predict(
            n=len(y_test),
            series=y_train_valid_scaled,
            future_covariates=X_all,
        )
    elif model_name in PAST_COV_MODELS:
        final_model.fit(
            series=y_train_valid_scaled,
            past_covariates=X_train_valid_all,
            verbose=False,
        )
        forecast_test_scaled = final_model.predict(
            n=len(y_test),
            series=y_train_valid_scaled,
            past_covariates=X_all,
        )
    else:
        raise ValueError(
            f"Unknown covariate usage for model '{model_name}'. "
            f"Please update FUTURE_COV_MODELS/PAST_COV_MODELS."
        )

    # Inverse-transform forecast to original scale
    y_test_forecast = y_scaler.inverse_transform(forecast_test_scaled)
    
    # ----------------------------------------------------------
    # 6) Save forecast series to CSV
    # ----------------------------------------------------------
    forecast_df = pd.DataFrame({
        "time": y_test.time_index,
        "y_actual": y_test.values().flatten(),
        "y_pred": y_test_forecast.values().flatten(),
    })
    forecast_path = os.path.join(output_dir, "forecast.csv")
    forecast_df.to_csv(forecast_path, index=False)

    # ----------------------------------------------------------
    # 7) Compute metrics
    # ----------------------------------------------------------
    metrics = {
        "model": model_name,
        "rmse": float(rmse(y_test, y_test_forecast)),
        "mae": float(mae(y_test, y_test_forecast)),
        "mape": float(mape(y_test, y_test_forecast)),
        "r2_score": float(r2_score(y_test, y_test_forecast)),
    }

    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(output_dir, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    logger.info("Evaluation metrics:")
    logger.info(metrics_df.to_string(index=False))
    
    # ----------------------------------------------------------
    # 8) Plot forecast series
    # ----------------------------------------------------------
    
    print("\n\n====================================================================================")
    print(pd.DataFrame(metrics_df))
    print("====================================================================================\n\n")
    
    plt.figure(figsize=(12, 5))   
    plt.plot(y_test.time_index, y_test.values(), label = 'Actual', color = 'orange')
    plt.plot(y_test.time_index, y_test_forecast.values(), label =f'{model_name} Forecast', linestyle='--')    
    plt.legend()
    plt.title(f'{model_name} Forecast Plot')
    plt.grid(True)
    plt.show()

    return {
        "metrics": metrics,
        "metrics_path": metrics_path,
        "best_params_path": best_params_path,
        "forecast_path": forecast_path,
    }




