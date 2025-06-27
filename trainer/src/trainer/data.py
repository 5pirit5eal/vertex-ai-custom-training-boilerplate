import json
import logging
import os
import time
from fnmatch import fnmatch
from typing import Any, Literal

import pandas as pd
from google.cloud import aiplatform, bigquery, storage
from numpy import inf
from sklearn.metrics import roc_curve

from trainer.config import Config


def get_bigquery_client(config: Config) -> bigquery.Client:
    """Creates a BigQuery client using the project ID from the config."""
    return bigquery.Client(project=config.project_id, location=config.region)


def get_gcs_client(config: Config) -> storage.Client:
    """Creates a Google Cloud Storage client."""
    return storage.Client(project=config.project_id)


def convert_gs_to_gcs(uri: str) -> str:
    """Converts a Google Cloud Storage URI to a Google Cloud Storage FUSE path."""
    uri = uri.strip()
    if uri.startswith("gs://"):
        return "/gcs/" + uri[5:]
    else:
        raise ValueError(f"Invalid GCS URI: {uri}")


def load_split_df(
    config: Config, split: Literal["train", "val", "test"]
) -> pd.DataFrame:
    """Loads data for the specified split from the URI in the config.
    As URIs to jsonl or csv files are in wildcard format, we need to
    convert them to a GCS FUSE path.
    If the data format is BigQuery, we use the BigQuery client to load the data.

    Args:
        config (Config): The configuration object containing the data URI and format.
        split (Literal["train", "val", "test"]): The data split to load.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    match split:
        case "train":
            data_uri = config.train_data_uri
        case "val":
            data_uri = config.val_data_uri
        case "test":
            data_uri = config.test_data_uri
        case _:
            raise ValueError(f"Invalid split: {split}")

    if config.data_format == "bigquery":
        logging.info(f"Loading {split} data from BigQuery")
        client = get_bigquery_client(config)
        query = f"SELECT * FROM `{data_uri}` WHERE split = '{split}'"
        df = client.query(query).to_dataframe()
    elif config.data_format == "csv":
        logging.info(f"Loading {split} csv data from GCS")
        # Load data from GCS
        try:
            uri = convert_gs_to_gcs(data_uri)

            df = pd.read_csv(uri)
        except Exception:
            df = load_wildcard_csv(config, data_uri)
    else:
        logging.info(f"Loading {split} jsonl data from GCS")
        raise ValueError(
            "JSONL format is not supported yet, and not used for Tabular Datasets."
        )
    return df.reset_index(drop=True)


def load_wildcard_csv(config: Config, uri: str) -> pd.DataFrame:
    """Loads data from a GCS URI with wildcard support.

    The URI is expected in the format `gs://bucket_name/path/training-*`

    Args:
        config (Config): The configuration object containing the data URI.
        uri (str): The GCS URI to load data from.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    # Use the Google Cloud Storage client to list files matching the wildcard
    client = get_gcs_client(config)

    # Get the bucket name and prefix from the URI
    bucket_name, prefix = uri[5:].split("/", 1)
    bucket = client.bucket(bucket_name)
    blobs: list[storage.Blob] = bucket.list_blobs()

    # Load each CSV file into a DataFrame and concatenate them
    return pd.concat(
        [
            pd.read_csv(convert_gs_to_gcs(f"gs://{bucket.name}/{blob.name}"))
            for blob in blobs
            if fnmatch(blob.name, prefix)
        ],
        ignore_index=True,
    )


def load_data(
    config: Config,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """Loads the training, validation, and test data from the URIs in the config.

    Args:
        config (Config): The configuration object containing the data URIs.

    Returns:
        The loaded data as a pandas DataFrames.
    """
    train_df = load_split_df(config, "train")

    val_df = (
        load_split_df(config, "val")
        if config.val_data_uri is not None
        else None
    )

    if any(
        preset in config.presets
        for preset in ["best_quality", "good_quality", "high_quality"]
    ):
        # Concatenate train and val data for training
        train_df = (
            pd.concat([train_df, val_df], ignore_index=True).reset_index(
                drop=True
            )
            if val_df is not None
            else train_df
        )
        val_df = None

    test_df = (
        load_split_df(config, "test")
        if config.test_data_uri is not None
        else None
    )

    # Combine the data into a single DataFrame
    return train_df, val_df, test_df


def write_df(
    config: Config,
    df: pd.DataFrame,
    filename: str,
) -> None:
    """Writes a DataFrame to a GCS URI.

    Args:
        config (Config): The configuration object containing the data URI.
        df (pd.DataFrame): The DataFrame to write.
        filename (str): The filename to write to.
    """
    # Convert the GCS URI to a GCS FUSE path
    uri = convert_gs_to_gcs(config.tensorboard_log_uri)

    # Create the directory if it doesn't exist
    os.makedirs(uri, exist_ok=True)

    uri = uri + "/" + filename

    # Write the DataFrame to a CSV file in GCS
    df.to_csv(uri)


def write_json(
    config: Config,
    data: dict,
    filename: str,
) -> None:
    """Writes a dict as JSON to a GCS URI.
    Args:
        config (Config): The configuration object containing the data URI.
        data (dict): The dict to write.
        filename (str): The filename to write to.
    """
    # Convert the GCS URI to a GCS FUSE path
    uri = convert_gs_to_gcs(config.tensorboard_log_uri)

    # Create the directory if it doesn't exist
    os.makedirs(uri, exist_ok=True)

    uri = uri + "/" + filename

    # Write the dict to a JSON file in GCS
    with open(uri, "w") as f:
        json.dump(data, f, skipkeys=True)


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
    for model_name, data in model_data.items():
        learning_curves = data[2]
        splits = data[0]
        metrics = data[1]

        # learning curves are per metric, per split
        for n, metric in enumerate(metrics):
            for split, curve in zip(splits, learning_curves[n]):
                results[f"{model_name} {metric} {split}"] = curve

    # Get the maximum length of the curves
    max_length = max(len(curve) for curve in results.values())

    for n in range(max_length):
        aiplatform.log_time_series_metrics(
            {key: curve[n] for key, curve in results.items() if len(curve) > n},
            step=n,
        )
        time.sleep(0.1)  # Sleep to avoid rate limiting


def log_roc_curve(
    label_column: str,
    positive_class: str,
    test_df: pd.DataFrame,
    test_predictions: pd.DataFrame,
) -> None:
    """Logs the ROC curve for a binary classification problem to an aiplatform experiment.

    Args:
        label_column (str): The name of the column containing the true labels.
        positive_class (str): The class label for the positive class.
        test_df (pd.DataFrame): The DataFrame containing the test data.
        test_predictions (pd.DataFrame): The DataFrame containing the test predictions.
    """
    y_true_numerical = test_df[label_column].apply(
        lambda x: 1 if x == positive_class else 0
    )
    fpr, tpr, threshold = roc_curve(
        y_true_numerical, test_predictions[positive_class]
    )
    fpr[fpr == inf] = 1.0  # Replace inf with 1.0 for plotting
    tpr[tpr == inf] = 1.0  # Replace inf with 1.0 for plotting
    threshold[threshold == inf] = 1.0  # Replace inf with 1.0 for plotting

    aiplatform.log_classification_metrics(
        fpr=fpr.tolist(),
        tpr=tpr.tolist(),
        threshold=threshold.tolist(),
        display_name=f"ROC Curve - {positive_class}",
    )
