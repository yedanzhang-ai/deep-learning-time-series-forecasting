#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
tuning.py

Generic hyperparameter tuning utilities using Optuna for multiple Darts models:

- GRU
- LSTM
- N-BEATS
- TCN
- TiDE
- TFT

Key features:
- Hyperparameter search space is read from a YAML config (hp_space).
- Models are constructed via the central factory in models.py (build_model).
- Covariate usage (past vs future) is handled per model.
- Designed to be reused across pipelines for all supported models.
"""
import gc
import yaml
import torch
import optuna

from optuna.pruners import MedianPruner
from darts.metrics import mae
from pytorch_lightning.callbacks import EarlyStopping

from .models import build_model

# ---------------------------------------------------------------------------
# Covariate usage per model
# ---------------------------------------------------------------------------

# Models that use FUTURE covariates in Darts
FUTURE_COV_MODELS = {"gru", "lstm", "tide", "tft"}

# Models that use PAST covariates in Darts
PAST_COV_MODELS = {"nbeats", "tcn"}


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def load_model_config(config_path):
    """
    Load the model configuration from a YAML file.

    Args:
        config_path (str): Path to model YAML config.

    Returns:
        dict: The dictionary under top-level "model".
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg["model"]


def sample_hp_from_space(trial, hp_space):
    """
    Sample a hyperparameter dictionary from a YAML-defined search space.

    Args:
        trial (optuna.Trial): Current Optuna trial.
        hp_space (dict): Hyperparameter search space from YAML.

    Returns:
        dict: Sampled hyperparameter values.
    """
    hp = {}

    for name, spec in hp_space.items():
        # Case 1: categorical list
        if isinstance(spec, list):
            hp[name] = trial.suggest_categorical(name, spec)

        # Case 2: dict specifying type and range
        elif isinstance(spec, dict):
            ptype = spec.get("type", "float")
            low = spec.get("low")
            high = spec.get("high")

            if ptype in ("float", "log_float"):
                hp[name] = trial.suggest_float(name, float(low), float(high), log=(ptype == "log_float"))
                
            elif ptype in ("int", "log_int"):
                hp[name] = trial.suggest_int(name, int(low), int(high), log=(ptype == "log_int"))
            else:
                raise ValueError(f"Unsupported hp_space type '{ptype}' for parameter '{name}'.")
        else:
            raise ValueError(f"Unsupported hp_space spec for '{name}': {spec}")

    return hp


# ---------------------------------------------------------------------------
# Objective builder (generic for all models)
# ---------------------------------------------------------------------------

def build_objective(
    model_name,
    y_train_scaled,
    X_train_all,
    y_valid,
    X_train_valid_all,
    y_scaler,
    forecast_horizon,
    use_gpu,
    model_cfg,
):
    """
    Build a generic Optuna objective function using:

    - model_name (to choose covariate usage & factory),
    - hp_space from model_cfg["hp_space"],
    - training parameters from model_cfg (n_epochs_tune, seed, etc.),
    - centralized model factory build_model(...).

    Args:
        model_name (str): One of {"gru", "lstm", "nbeats", "tcn", "tide", "tft"}.
        y_train_scaled: Scaled training target TimeSeries.
        X_train_all: Covariates TimeSeries for the training period.
        y_valid: Validation target TimeSeries (unscaled).
        X_train_valid_all: Covariates TimeSeries for train+valid period.
        y_scaler: Fitted Scaler for target.
        forecast_horizon (int): Forecast horizon for predict / model construction.
        model_cfg (dict): Model config from YAML (`load_model_config`).

    Returns:
        callable: Objective function for Optuna.
    """
    name = model_name.lower()
    hp_space = model_cfg["hp_space"]
    n_epochs_tune = model_cfg["n_epochs_tune"]
    seed = model_cfg['seed']    
    early_stop_patience = model_cfg['early_stop_patience']

    # EarlyStopping callback (reused across trials)
    early_stop = EarlyStopping(
        monitor="train_loss",
        mode="min",
        patience=early_stop_patience,
        min_delta=1e-4,
        verbose=False,
    )

    def objective(trial: optuna.Trial) -> float:
        """
        Generic Optuna objective function for this model.

        Steps:
        - Sample hp from YAML-defined space.
        - Build model via models.build_model(...).
        - Fit model on training data with correct covariate type.
        - Predict on (train+valid) horizon for validation window.
        - Compute MAE on unscaled y_valid.
        """
        try:
            torch.cuda.empty_cache()
            gc.collect()

            # 1) Sample hyperparameters
            hp = sample_hp_from_space(trial, hp_space)

            # 2) Build model via central factory
            model = build_model(
                model_name=name,
                hp=hp,
                n_epochs=n_epochs_tune,
                early_stop=early_stop,
                forecast_horizon=forecast_horizon,
                SEED=seed,
                use_gpu=use_gpu,
            )

            # 3) Fit with appropriate covariate type
            if name in FUTURE_COV_MODELS:
                model.fit(series=y_train_scaled, future_covariates=X_train_all, verbose=False)
                valid_pred_scaled = model.predict(
                    n=len(y_valid), 
                    series=y_train_scaled,
                    future_covariates=X_train_valid_all
                )
            elif name in PAST_COV_MODELS:
                model.fit(series=y_train_scaled, past_covariates=X_train_all, verbose=False)
                valid_pred_scaled = model.predict(
                    n=len(y_valid),
                    series=y_train_scaled,
                    past_covariates=X_train_valid_all
                )
            else:
                raise ValueError(
                    f"Unknown covariate usage for model '{model_name}'. "
                    f"Update FUTURE_COV_MODELS/PAST_COV_MODELS in tuning.py."
                )

            # 4) Inverse-transform predictions and compute MAE
            valid_pred = y_scaler.inverse_transform(valid_pred_scaled)
            score = mae(y_valid, valid_pred)

            # 5) Cleanup
            del model, valid_pred_scaled
            torch.cuda.empty_cache()
            gc.collect()

            return float(score)

        except Exception as e:
            torch.cuda.empty_cache()
            gc.collect()
            print(f"Trial failed ({model_name}): {e}")
            return 1e6

    return objective


# ---------------------------------------------------------------------------
# Public: tune_model_fn
# ---------------------------------------------------------------------------

def tune_model_fn(
    model_name,
    y_train_scaled,
    X_train_all,
    y_valid,
    X_train_valid_all,
    y_scaler,
    forecast_horizon,
    use_gpu,
    config_path,
):
    """
    Run Optuna hyperparameter tuning for a given model using YAML-configured
    hyperparameter space and training parameters.

    Args:
        model_name (str): Model key: "gru", "lstm", "nbeats", "tcn", "tide", "tft".
        y_train_scaled: Scaled training target series (TimeSeries).
        X_train_all: Covariates TimeSeries for the training period (scaled + non-scaled).
        y_valid: Validation target series (unscaled).
        X_train_valid_all: Covariates TimeSeries for train+valid period.
        y_scaler: Target scaler (to invert predictions).
        forecast_horizon (int): Forecast horizon for the model.
        config_path (str, optional): Path to model YAML config. 

    Returns:
        optuna.Study: Completed Optuna study (with best_params, best_value, etc.).
    """
   
    name = model_name.lower()
    if config_path is None:
        config_path = f"config/model_{name}.yaml"

    # Load YAML model config
    model_cfg = load_model_config(config_path)

    # Extract tuning parameters
    n_trials = model_cfg["n_trials"]
    seed = model_cfg["seed"]

    # Build Optuna objective
    objective = build_objective(
        model_name=name,
        y_train_scaled=y_train_scaled,
        X_train_all=X_train_all,
        y_valid=y_valid,
        X_train_valid_all=X_train_valid_all,
        y_scaler=y_scaler,
        forecast_horizon=forecast_horizon,
        use_gpu=use_gpu,
        model_cfg=model_cfg,
    )

    # Create sampler & pruner
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    # Run study
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner,)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study

