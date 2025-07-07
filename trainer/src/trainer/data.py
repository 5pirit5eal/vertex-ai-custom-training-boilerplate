import json
import logging
import os
import time
from fnmatch import fnmatch
from typing import Any, Literal

import pandas as pd
import yaml
from autogluon.tabular import TabularPredictor
from autogluon.tabular.version import __version__ as autogluon_version
from google.cloud import aiplatform, bigquery, storage
from numpy import inf, linspace
from sklearn.metrics import roc_curve

from trainer.config import Config


def get_bigquery_client(config: Config) -> bigquery.Client:
    """Creates a BigQuery client using the project ID from the config."""
    return bigquery.Client(project=config.project_id, location=config.region)


def get_gcs_client(config: Config) -> storage.Client:
    """Creates a Google Cloud Storage client."""
    return storage.Client(project=config.project_id)


def gcs_path(uri: str, *path_segments: str) -> str:
    """Converts a Google Cloud Storage URI to a Google Cloud Storage FUSE path.

    Args:
        uri (str): The GCS URI to convert (e.g., "gs://bucket/path").
        *path_segments (str): Optional path segments to join to the converted path.

    Returns:
        str: The GCS FUSE path with any additional path segments joined.

    Example:
        gcs_path("gs://my-bucket/data") -> "/gcs/my-bucket/data"
        gcs_path("gs://my-bucket/data", "file.csv") -> "/gcs/my-bucket/data/file.csv"
        gcs_path("gs://my-bucket", "folder", "subfolder", "file.txt") -> "/gcs/my-bucket/folder/subfolder/file.txt"
    """
    uri = uri.strip()
    if not uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {uri}")

    # Convert gs:// to /gcs/
    base_path = "/gcs/" + uri[5:]

    # Join any additional path segments
    if path_segments:
        return os.path.join(base_path, *path_segments)

    return base_path


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

    if data_uri is None:
        raise ValueError(f"No {split} data URI provided, cannot load data.")

    if config.data_format == "bigquery":
        logging.info(f"Loading {split} data from BigQuery")
        client = get_bigquery_client(config)
        query = f"SELECT * FROM `{data_uri}` WHERE split = '{split}'"
        df = client.query(query).to_dataframe()
    elif config.data_format == "csv":
        logging.info(f"Loading {split} csv data from GCS")
        # Load data from GCS
        try:
            uri = gcs_path(data_uri)

            df = pd.read_csv(uri)
        except Exception:
            df = load_wildcard_csv(config, data_uri)
    else:
        logging.info(f"Loading {split} jsonl data from GCS")
        # TODO: Implement JSONL loading
        raise ValueError(
            "JSONL format is not supported yet, and not used for Vertex AI Tabular Datasets. "
            "Checkout multimodal training in tabular format on https://auto.gluon.ai."
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
            pd.read_csv(gcs_path(f"gs://{bucket.name}/{blob.name}"))
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
    # Convert the GCS URI to a GCS FUSE path and join with filename
    uri = gcs_path(config.tensorboard_log_uri, filename)

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(uri), exist_ok=True)

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
    # Convert the GCS URI to a GCS FUSE path and join with filename
    uri = gcs_path(config.tensorboard_log_uri, filename)

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(uri), exist_ok=True)

    # Write the dict to a JSON file in GCS
    try:
        with open(uri, "w") as f:
            json.dump(data, f, skipkeys=True)
    except Exception as e:
        logging.error(f"Error writing JSON to {uri}: {e}", exc_info=e)
        return


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
    config: Config, predictor: TabularPredictor, prefix: str = ""
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
    try:
        y_true_numerical = test_df[label_column].apply(
            lambda x: 1 if x == positive_class else 0
        )
        fpr, tpr, threshold = roc_curve(
            y_true_numerical, test_predictions[positive_class]
        )
        fpr[fpr == inf] = 1.0  # Replace inf with 1.0 for plotting
        tpr[tpr == inf] = 1.0  # Replace inf with 1.0 for plotting
        threshold[threshold == inf] = 1.0  # Replace inf with 1.0 for plotting

        # Subsample the data to 1000 points for plotting
        if len(fpr) > 1000:
            indices = linspace(0, len(fpr) - 1, 1000, dtype=int)
            fpr = fpr[indices]
            tpr = tpr[indices]
            threshold = threshold[indices]

        aiplatform.log_classification_metrics(
            fpr=fpr.tolist(),
            tpr=tpr.tolist(),
            threshold=threshold.tolist(),
            display_name=f"ROC Curve - {positive_class}",
        )
    except Exception as e:
        logging.error(f"Error logging ROC curve: {e}", exc_info=e)
        return


def log_container_execution(config: Config) -> None:
    """Logs the execution of a container to an aiplatform experiment.

    Args:
        config (Config): The configuration object containing the model export URI and label.
    """
    # TODO: Differentiate between Vertex Dataset and Custom Dataset
    #       and use the correct schema for the input artifacts.
    #       Currently, we use the system.Dataset schema for both.
    input_artifacts = [
        aiplatform.Artifact.create(
            schema_title="system.Dataset",
            uri=split[1],
            metadata=dict(
                payload_format=config.data_format.upper(),
            ),
            display_name=f"{split[0]} data for {config.label}",
        )
        for split in zip(
            ["Train", "Validation", "Test"],
            [config.train_data_uri, config.val_data_uri, config.test_data_uri],
        )
        if split[1] is not None
    ]
    model_artifact = aiplatform.Artifact.create(
        schema_title="system.Model",
        uri=config.model_export_uri,
        display_name=f"AutoGluon model for {config.label}",
        metadata=dict(
            framework_version=autogluon_version,
            framework="AutoGluon",
        ),
    )
    with aiplatform.start_execution(
        schema_title="system.ContainerExecution",
        display_name="AutoGluon Training",
    ) as execution:
        # Log the execution
        execution.assign_input_artifacts(input_artifacts)
        execution.assign_output_artifacts([model_artifact])


def convert_feature_map_to_schema(
    map: dict[str, tuple[str, tuple]],
) -> dict[str, Any]:
    type_map: dict[str, tuple[str, str | None]] = {
        "int": ("integer", "int64"),
        "float": ("number", "float"),
        "object": ("string", None),
        "category": ("string", None),
        "bool": ("boolean", None),
        "datetime": ("string", "date-time"),
        "text": ("string", None),
    }
    properties = {}
    required = []
    for feature, dtype in map.items():
        openapi_type, openapi_format = type_map.get(dtype[0], ("string", None))
        prop = {"type": openapi_type}
        if openapi_format:
            prop["format"] = openapi_format
        properties[feature] = prop
        required.append(feature)
    schema = {
        "type": "object",
        "properties": properties,
        "required": required,
    }
    return schema


def convert_class_labels_to_schema(
    class_labels: list[str | int],
) -> dict[str, Any]:
    """Converts class labels to a schema for OpenAPI 3.0.2, matching a response of
    a list of dicts where each dict has class labels as keys and probabilities as values.

    Args:
        class_labels (list[str | int]): The list of class labels.

    Returns:
        dict[str, Any]: The schema for the class labels.
    """
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                str(label): {"type": "number"} for label in class_labels
            },
            "required": [str(label) for label in class_labels],
            "additionalProperties": False,
        },
    }


def write_instance_and_prediction_schemas(
    config: Config, predictor: TabularPredictor
) -> None:
    """Creates and saves the parameters and results schema for the model
    as YAML-files.

    Args:
        config (Config): The configuration object containing the model export URI.
        predictor (TabularPredictor): The predictor object containing the model parameters and results.
    """
    try:
        feature_metadata_dict = predictor.feature_metadata_in.to_dict()
        classes = predictor.class_labels

        feature_schema = convert_feature_map_to_schema(feature_metadata_dict)
        label_schema = convert_class_labels_to_schema(classes)

        # Write the schema to a YAML file
        for schema, filename in [
            (feature_schema, "instance_schema.yaml"),
            (label_schema, "prediction_schema.yaml"),
        ]:
            schema_path = gcs_path(config.model_export_uri, filename)
            os.makedirs(os.path.dirname(schema_path), exist_ok=True)
            with open(schema_path, "w") as f:
                yaml.dump(schema, f, sort_keys=False)
    except Exception as e:
        logging.error(f"Error writing schemas: {e}", exc_info=e)
        return
