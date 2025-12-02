#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
models.py

Model factory utilities for Darts deep learning models used in this project:

- GRU      (RNNModel with GRU architecture)
- LSTM     (RNNModel with LSTM architecture)
- N-BEATS  (NBEATSModel)
- TCN      (TCNModel)
- TiDE     (TiDEModel)
- TFT      (TFTModel)

Each builder:
- Takes a hyperparameter dict `hp` (from Optuna or config),
- Uses a forecast horizon for output_chunk_length,
- Shares consistent PyTorch Lightning trainer kwargs,
- Uses L1Loss by default (robust to outliers, good for many forecasting tasks).

These functions are designed to be used in:
- Hyperparameter tuning objectives (Optuna),
- Final training pipelines for each model.
"""

import logging
import warnings

from torch.nn import L1Loss
from pytorch_lightning.callbacks import EarlyStopping

from darts.models import (
    RNNModel,
    TCNModel,
    NBEATSModel,
    TFTModel,
    TiDEModel,
)


# ---------------------------------------------------------------------------
# Logging & warnings configuration
# ---------------------------------------------------------------------------

logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.ERROR)
logging.getLogger("darts").setLevel(logging.ERROR)

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper: build shared pl_trainer_kwargs
# ---------------------------------------------------------------------------

def _build_pl_trainer_kwargs(early_stop, n_epochs, use_gpu):
    """
    Build a consistent set of PyTorch Lightning trainer kwargs for all models.

    Args:
        early_stop (EarlyStopping): EarlyStopping callback to control overfitting during tuning/training.
        n_epochs (int): Upper bound on epochs (also passed to the model as n_epochs).
        use_gpu (bool): Whether to use GPU ("gpu") or CPU ("cpu") accelerator.

    Returns:
        dict: Keyword arguments to pass via `pl_trainer_kwargs` to Darts models.
    """
    return {
        "accelerator": "gpu" if use_gpu else "cpu",
        "devices": 1,
        "precision": "32-true",          
        "enable_progress_bar": False,
        "logger": False,
        "enable_model_summary": False,
        "max_epochs": n_epochs,
        "callbacks": [early_stop],
    }

# ---------------------------------------------------------------------------
# Individual model builders
# ---------------------------------------------------------------------------


def build_gru_model_fn(hp, n_epochs, early_stop, forecast_horizon, SEED, use_gpu):
    """
    Build a Darts RNNModel with GRU architecture.

    Args:
        hp (dict): Hyperparameters, expected keys:
                - "lookback"
                - "hidden"
                - "layers"
                - "dropout"
                - "batch"
                - "lr"
                - "weight_decay"
        n_epochs (int): Number of training epochs.
        early_stop (EarlyStopping):Early stopping callback.
        forecast_horizon (int): Forecast horizon (output_chunk_length).
        SEED (int): Random seed for reproducibility.
        use_gpu (bool): Whether to use GPU ("gpu") or CPU ("cpu").

    Returns:
        RNNModel: Configured GRU-based RNNModel instance.
    """
    pl_trainer_kwargs = _build_pl_trainer_kwargs(early_stop, n_epochs, use_gpu)
    
    kwargs = dict(
        model="GRU",
        input_chunk_length=hp["lookback"],   
        output_chunk_length=forecast_horizon,
        training_length = hp["lookback"] + forecast_horizon*2,
        n_rnn_layers=hp['layers'],  
        hidden_dim=hp["hidden"],        
        dropout=hp["dropout"],
        batch_size=hp["batch"],       
        optimizer_kwargs={
            "lr": hp["lr"],
            "weight_decay": hp["weight_decay"]
        },      
        random_state=SEED,
        pl_trainer_kwargs=pl_trainer_kwargs
    )

    return RNNModel(n_epochs=n_epochs, **kwargs)

def build_lstm_model_fn(hp, n_epochs, early_stop, forecast_horizon, SEED, use_gpu):
    """
    Build a Darts RNNModel with LSTM architecture using given hyperparameters.

    Args:
        hp (dict): Hyperparameter dictionary with keys:
            - "lookback"
            - "hidden"
            - "layers"
            - "dropout"
            - "batch"
            - "lr"
            - "weight_decay"
        n_epochs (int): Number of training epochs.
        early_stop(EarlyStopping): Early stopping callback.
        forecast_horizon (int): Forecast horizon.
        SEED (int): Random seed for reproducibility.
        use_gpu (bool): Whether to use GPU ("gpu") or CPU ("cpu").

    Returns:
        RNNModel: Configured LSTM-based RNNModel instance.
    """
    pl_trainer_kwargs = _build_pl_trainer_kwargs(early_stop, n_epochs, use_gpu)
    
    kwargs = dict(
        model="LSTM",
        input_chunk_length=hp["lookback"],   
        output_chunk_length=forecast_horizon,
        training_length = hp["lookback"] + forecast_horizon*2,
        n_rnn_layers=hp['layers'],  
        hidden_dim=hp["hidden"],        
        dropout=hp["dropout"],
        batch_size=hp["batch"],  
        optimizer_kwargs={
            "lr": hp["lr"],
            "weight_decay": hp["weight_decay"]},      
        loss_fn = L1Loss(),
        random_state=SEED,
        pl_trainer_kwargs=pl_trainer_kwargs
    )

    return RNNModel(n_epochs=n_epochs, **kwargs)

def build_nbeats_model_fn(hp, n_epochs, early_stop, forecast_horizon, SEED, use_gpu):
    """
    Build a Darts NBEATSModel using given hyperparameters.

    Args:
        hp (dict): Hyperparameters, expected keys:
                - "lookback"
                - "stacks"
                - "blocks"
                - "width"
                - "dropout"
                - "batch"
                - "lr"
                - "weight_decay"
        n_epochs (int): Number of training epochs.
        early_stop (EarlyStopping): Early stopping callback.
        forecast_horizon (int):Forecast horizon (output_chunk_length).
        SEED (int): Random seed.
        use_gpu (bool): Whether to use GPU or CPU.

    Returns:
        NBEATSModel: Configured N-BEATS model instance.
    """
    pl_trainer_kwargs = _build_pl_trainer_kwargs(early_stop, n_epochs, use_gpu)

    kwargs = dict(      
        input_chunk_length=hp["lookback"],   
        output_chunk_length=forecast_horizon,      
        generic_architecture=True,         
        num_stacks=hp["stacks"],
        num_blocks=hp["blocks"],
        layer_widths=hp["width"],        
        dropout=hp["dropout"],       
        batch_size=hp["batch"],       
        optimizer_kwargs={
            "lr": hp["lr"],
            "weight_decay": hp["weight_decay"]},      
        loss_fn = L1Loss(),
        random_state=SEED,
        pl_trainer_kwargs=pl_trainer_kwargs
    )

    return NBEATSModel(n_epochs=n_epochs, **kwargs)

def build_tcn_model_fn(hp, n_epochs, early_stop, forecast_horizon, SEED, use_gpu):
    """
    Build a Darts TCNModel using given hyperparameters.

    Args:
        hp (dict): Hyperparameters, expected keys:
                - "lookback"
                - "filters"
                - "kernel"
                - "dilation_base"
                - "dropout"
                - "batch"
                - "lr"
                - "weight_decay"
        n_epochs (int): Number of training epochs.
        early_stop (EarlyStopping): Early stopping callback.
        forecast_horizon (int): Forecast horizon (output_chunk_length).
        SEED (int): Random seed.
        use_gpu (bool): Whether to use GPU or CPU.

    Returns:
        TCNModel: Configured Temporal Convolutional Network model.
    """
    pl_trainer_kwargs = _build_pl_trainer_kwargs(early_stop, n_epochs, use_gpu)
    
    kwargs = dict(      
        input_chunk_length=hp["lookback"],   
        output_chunk_length=forecast_horizon,
        num_filters=hp["filters"],
        kernel_size=hp["kernel"],
        dilation_base=hp["dilation_base"],              
        dropout=hp["dropout"],
        weight_norm=False,
        batch_size=hp["batch"],       
        optimizer_kwargs={
            "lr": hp["lr"],
            "weight_decay": hp["weight_decay"]
        },      
        loss_fn = L1Loss(),
        random_state=SEED,
        pl_trainer_kwargs=pl_trainer_kwargs
    )

    return TCNModel(n_epochs=n_epochs, **kwargs)

def build_tft_model_fn(hp, n_epochs, early_stop, forecast_horizon, SEED, use_gpu):
    """
    Build a Darts TFTModel (Temporal Fusion Transformer) using given hyperparameters.

    Args:
        hp (dict): Hyperparameters, expected keys:
                - "lookback"
                - "hidden"
                - "lstm_layers"
                - "heads"
                - "dropout"
                - "batch"
                - "lr"
                - "weight_decay"
        n_epochs (int): Number of training epochs.
        early_stop (EarlyStopping): Early stopping callback.
        forecast_horizon (int):Forecast horizon.
        SEED (int): Random seed.
        use_gpu (bool): Whether to use GPU or CPU.

    Returns:
        TFTModel: Configured Temporal Fusion Transformer model.
    """
    pl_trainer_kwargs = _build_pl_trainer_kwargs(early_stop, n_epochs, use_gpu)
    
    kwargs = dict(      
        input_chunk_length=hp["lookback"],   
        output_chunk_length=forecast_horizon,
        hidden_size=hp['hidden'],       
        lstm_layers=hp["lstm_layers"],
        num_attention_heads=hp["heads"],
        add_relative_index=True,
        dropout=hp["dropout"],       
        batch_size=hp["batch"],       
        optimizer_kwargs={
            "lr": hp["lr"],
            "weight_decay": hp["weight_decay"]
        },
        loss_fn = L1Loss(),
        random_state=SEED,
        pl_trainer_kwargs=pl_trainer_kwargs
    )

    return TFTModel(n_epochs=n_epochs, **kwargs)

def build_tide_model_fn(hp, n_epochs, early_stop, forecast_horizon, SEED, use_gpu):
    """
    Build a Darts TiDEModel using given hyperparameters.

    Args:
        hp (dict): Hyperparameters, expected keys:
                - "lookback"
                - "hidden"
                - "encoder_layers"
                - "decoder_layers"
                - "decoder_dim"
                - "temporal_width"
                - "dropout"
                - "batch"
                - "lr"
                - "weight_decay"
        n_epochs (int): Number of training epochs.
        early_stop (EarlyStopping): Early stopping callback.
        forecast_horizon (int): Forecast horizon.
        SEED (int): Random seed.
        use_gpu (bool):Whether to use GPU or CPU.

    Returns:
        TiDEModel: Configured TiDE model instance.
    """
    pl_trainer_kwargs = _build_pl_trainer_kwargs(early_stop, n_epochs, use_gpu)
    
    kwargs = dict(      
        input_chunk_length=hp["lookback"],   
        output_chunk_length=forecast_horizon,        
        num_encoder_layers=hp["encoder_layers"],
        num_decoder_layers=hp["decoder_layers"],
        decoder_output_dim=hp["decoder_dim"],
        hidden_size=hp["hidden"],
        temporal_width_past=hp["temporal_width"],
        temporal_width_future=hp["temporal_width"],
        dropout=hp["dropout"],       
        batch_size=hp["batch"],       
        optimizer_kwargs={
            "lr": hp["lr"],
            "weight_decay": hp["weight_decay"]
        },      
        loss_fn = L1Loss(),
        random_state=SEED,
        pl_trainer_kwargs=pl_trainer_kwargs
    )

    return TiDEModel(n_epochs=n_epochs, **kwargs)


# ---------------------------------------------------------------------------
# Model registry for tuning/pipelines
# ---------------------------------------------------------------------------

model_builders = {
    "gru": build_gru_model_fn,
    "lstm": build_lstm_model_fn,
    "nbeats": build_nbeats_model_fn,
    "tcn": build_tcn_model_fn,
    "tide": build_tide_model_fn,
    "tft": build_tft_model_fn,
}


def build_model(model_name, hp, n_epochs, early_stop, forecast_horizon, SEED, use_gpu):
    """
    Generic model factory: build a model by name using the appropriate builder.

    Args:
        model_name (str): One of: "gru", "lstm", "nbeats", "tcn", "tide", "tft".
        hp (dict): Hyperparameter dictionary.
        n_epochs (int): Number of epochs for training.
        early_stop (EarlyStopping): Early stopping callback.
        forecast_horizon (int): Forecast horizon for the model.
        SEED (int): Random seed.
        use_gpu (bool): Whether to use GPU.

    Returns:
        A configured Darts model instance.
    """
    name = model_name.lower()
    if name not in model_builders:
        raise ValueError(f"Unknown model_name '{model_name}'. "
                         f"Available: {list(model_builders.keys())}")

    builder = model_builders[name]
    return builder(
        hp=hp,
        n_epochs=n_epochs,
        early_stop=early_stop,
        forecast_horizon=forecast_horizon,
        SEED=SEED,
        use_gpu=use_gpu,
    )

