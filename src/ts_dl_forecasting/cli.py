#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
cli.py

Command-line entrypoint for ts_dl_forecasting.

Supports:
- Running a single experiment:
    python -m ts_dl_forecasting.cli --experiment config/experiment_lstm.yaml

- Running multiple experiments in batch:
    python -m ts_dl_forecasting.cli --batch config/experiments_batch.yaml

- Aggregating metrics & plotting all forecasts:
    python -m ts_dl_forecasting.cli --batch config/experiments_batch.yaml --report
"""

import argparse
import logging
import yaml
import os

from .pipeline import run_experiment
from .reporting import load_metrics, load_forecasts, plot_all_forecasts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ts_dl_forecasting CLI")

    parser.add_argument(
        "--experiment",
        type=str,
        help="Path to a single experiment YAML (e.g., config/experiment_lstm.yaml)",
    )
    parser.add_argument(
        "--batch",
        type=str,
        help="Path to a batch YAML listing experiment configs (e.g., config/experiments_batch.yaml)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="After batch, aggregate metrics and plot all forecasts.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs",
        help="Root directory where per-model outputs are saved (default: outputs/).",
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default=None,
        help="Optional path to save the comparison plot (PNG). If omitted, shows the plot interactively.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Mode 1: single experiment
    # ------------------------------------------------------------------
    if args.experiment and not args.batch:
        logger.info(f"Running single experiment: {args.experiment}")
        run_experiment(args.experiment)
        return

    # ------------------------------------------------------------------
    # Mode 2: batch experiments
    # ------------------------------------------------------------------
    if args.batch and not args.experiment:
        logger.info(f"Running batch experiments from: {args.batch}")

        with open(args.batch, "r") as f:
            batch_cfg = yaml.safe_load(f)

        exp_entries = batch_cfg["experiments"]
        model_names = []

        for entry in exp_entries:
            exp_path = entry["config"]
            # Read model_name from each experiment config so reporting knows which models
            with open(exp_path, "r") as ef:
                exp_cfg = yaml.safe_load(ef)
            model_name = exp_cfg["experiment"]["model_name"].lower()
            model_names.append(model_name)

            logger.info(f"  → Running experiment for model: {model_name} ({exp_path})")
            run_experiment(exp_path)

        # Optional reporting step
        if args.report:
            logger.info("Generating aggregated metrics and comparison plot ...")
            metrics_df = load_metrics(args.output_root, model_names)
            print("\nCombined metrics table:")
            print(metrics_df.to_string(index=False))

            forecasts = load_forecasts(args.output_root, model_names)
            plot_all_forecasts(
                forecasts,
                title="All models vs actual",
                save_path=args.plot_path,
            )

        return

    # ------------------------------------------------------------------
    # Invalid usage
    # ------------------------------------------------------------------
    raise SystemExit(
        "Please provide either --experiment or --batch.\n"
        "Examples:\n"
        "  python -m ts_dl_forecasting.cli --experiment config/experiment_lstm.yaml\n"
        "  python -m ts_dl_forecasting.cli --batch config/experiments_batch.yaml --report"
    )


if __name__ == "__main__":
    main()


# In[ ]:




