"""This module provides functions to log experiment results to Google Cloud Vertex AI Experiments."""

import logging
import time
from typing import Any

import pandas as pd
from autogluon.tabular import TabularPredictor
from google.cloud import aiplatform


from trainer.config import Config
from trainer.data import write_json
from trainer.vertex import calculate_roc_curve


def log_nested_metrics(metrics: dict[str, Any], prefix: str = "") -> None:
    """Recursively logs metrics to an aiplatform experiment."""
    for key, value in metrics.items():
        if isinstance(value, dict):
            if prefix:
                log_nested_metrics(value, prefix=f"{prefix} {key}")
            else:
                log_nested_metrics(value, prefix=key)
        else:
            if prefix:
                key = f"{prefix} {key}"
            aiplatform.log_metrics({key: value})


def log_learning_curves(model_data: dict[str, list]) -> None:
    """Logs learning curves returned by the predictor to an aiplatform experiment.

    Args:
        model_data (dict[str, list]): The model data containing the learning curves per model.
    """
    # Create a flattened dictionary of results
    # The keys will be in the format "model_name metric split"
    # The values will be the learning curves for that metric and split
    # e.g. "model_name accuracy train" -> [0.1, 0.2, 0.3, ...]
    results: dict[str, list] = {}
    try:
        for model_name, data in model_data.items():
            learning_curves = data[2]
            splits = data[0]
            metrics = data[1]

            # learning curves are per metric, per split
            for n, metric in enumerate(metrics):
                for split, curve in zip(splits, learning_curves[n]):
                    results[f"{model_name} {metric} {split}"] = curve

        # Get the maximum length of the curves
        max_length = max((len(curve) for curve in results.values()), default=0)

        for n in range(max_length):
            aiplatform.log_time_series_metrics(
                {
                    key: curve[n]
                    for key, curve in results.items()
                    if len(curve) > n
                },
                step=n,
            )
            time.sleep(0.1)  # Sleep to avoid rate limiting
    except Exception as e:
        logging.error(f"Error logging learning curves: {e}")
        return


def log_metadata(
    config: Config, predictor: TabularPredictor, prefix: str
) -> None:
    """Logs metadata about the run to GCS."""
    logging.info("Writing metadata and learning curves to GCS...")
    summary = predictor.fit_summary(show_plot=False)
    del summary["leaderboard"]
    write_json(config, summary, f"{prefix}_fit_summary.json")

    metadata, model_data = predictor.learning_curves()
    write_json(config, data=metadata, filename=f"{prefix}_metadata.json")
    write_json(config, data=model_data, filename=f"{prefix}_model_data.json")

    if config.experiment_name:
        if model_data:
            logging.info(f"Logging {prefix} learning curves to Vertex AI...")
            log_learning_curves(model_data)
        else:
            logging.info(f"No {prefix} learning curves to log.")


def log_roc_curve(
    label_column: str,
    positive_class: str | int,
    df: pd.DataFrame,
    predictions: pd.DataFrame,
    prefix: str,
) -> None:
    """Logs the ROC curve for a binary classification problem to an aiplatform experiment.

    Args:
        label_column (str): The name of the column containing the true labels.
        positive_class (str): The class label for the positive class.
        df (pd.DataFrame): The DataFrame containing the test data.
        predictions (pd.DataFrame): The DataFrame containing the test predictions.
    """
    try:
        fpr, tpr, threshold = calculate_roc_curve(
            label_column, positive_class, df, predictions
        )

        aiplatform.log_classification_metrics(
            fpr=fpr.tolist(),
            tpr=tpr.tolist(),
            threshold=threshold.tolist(),
            display_name=f"ROC Curve - {prefix.upper()} - {positive_class}",
        )
    except Exception as e:
        logging.error(
            f"Error logging {prefix.upper()} ROC curve: {e}", exc_info=e
        )
        return
