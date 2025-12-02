#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
reporting.py

Tools to:

1. Aggregate metrics from multiple model output directories
   into a single pandas DataFrame.

2. Plot y_test and forecasts from multiple models together
   on one figure for visual comparison.

Assumes each model's output_dir contains:
- metrics.csv
- forecast.csv   (with columns: time, y_actual, y_pred)
"""
import os

import pandas as pd
import matplotlib.pyplot as plt


def load_metrics(output_root, model_names):
    """
    Load metrics.csv for multiple models and combine into one DataFrame.

    Args:
        output_root (str): Base directory (e.g., "outputs").
        model_names (list[str]): Model names like ["lstm", "gru", "tcn", ...].

    Returns:
        pd.DataFrame: Combined metrics table, one row per model.
    """
    dfs = []
    for name in model_names:
        path = os.path.join(output_root, name, "metrics.csv")
        if not os.path.exists(path):
            print(f"[WARNING] metrics.csv not found for model {name} at {path}")
            continue
        df = pd.read_csv(path)
        # Ensure the model column is correct / present
        df["model"] = name
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError("No metrics files found.")

    return pd.concat(dfs, ignore_index=True)


def load_forecasts(output_root, model_names):
    """
    Load forecast.csv for multiple models.

    Args:
        output_root (str): Base directory (e.g., "outputs").
        model_names (list[str]): Model names.

    Returns:
        dict[str, pd.DataFrame]: Mapping from model_name -> forecast DataFrame.
    """
    forecasts = {}
    for name in model_names:
        path = os.path.join(output_root, name, "forecast.csv")
        if not os.path.exists(path):
            print(f"[WARNING] forecast.csv not found for model {name} at {path}")
            continue
        df = pd.read_csv(path, parse_dates=["time"])
        forecasts[name] = df

    if not forecasts:
        raise FileNotFoundError("No forecast files found.")

    return forecasts


def plot_all_forecasts(forecasts, title, save_path):
    """
    Plot actual vs forecasts from multiple models on a single figure.

    Args:
        forecasts (dict): {model_name: forecast_df}
        title (str): Plot title.
        save_path (str | None): If provided, save the figure to this path.
    """
    if not forecasts:
        raise ValueError("No forecasts provided.")

    # Use the first model's DF to get the time and actual series
    first_model = next(iter(forecasts.keys()))
    base_df = forecasts[first_model]

    plt.figure(figsize=(12, 6))

    # Plot actual
    plt.plot(base_df["time"], base_df["y_actual"], label="Actual", linewidth=2)

    # Plot each model's forecast
    for model_name, df in forecasts.items():
        plt.plot(
            df["time"],
            df["y_pred"],
            label=model_name.upper(),
            linestyle="--",
            alpha=0.8,
        )

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
    else:
        plt.tight_layout()
        plt.show()

