import json
import logging
import os
from fnmatch import fnmatch
from typing import Any, Literal

import pandas as pd
import yaml
from autogluon.tabular import TabularPredictor
from google.cloud import bigquery, storage
from numpy import inf, linspace
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

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


def convert_feature_map_to_input_schema(
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


def convert_class_labels_to_pred_schema(
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
        "type": "object",
        "properties": {
            str(label): {"type": "number"} for label in class_labels
        },
        "required": [str(label) for label in class_labels],
        "additionalProperties": False,
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

        feature_schema = convert_feature_map_to_input_schema(
            feature_metadata_dict
        )
        label_schema = convert_class_labels_to_pred_schema(classes)

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


def calculate_roc_curve(
    label_column: str,
    positive_class: str | int,
    df: pd.DataFrame,
    predictions: pd.DataFrame,
):
    """Calculates the ROC curve for a binary classification problem.

    Args:
        label_column (str): The name of the column containing the true labels.
        positive_class (str | int): The class label for the positive class.
        df (pd.DataFrame): The DataFrame containing the test data.
        predictions (pd.DataFrame): The DataFrame containing the test predictions.
    Returns:
        tuple: A tuple containing the false positive rates, true positive rates, and thresholds.
    """
    y_true_numerical = df[label_column].apply(
        lambda x: 1 if x == positive_class else 0
    )
    fpr, tpr, threshold = roc_curve(
        y_true_numerical, predictions[positive_class]
    )
    # Replace inf with 1.0 for plotting
    fpr[fpr == inf] = 1.0
    tpr[tpr == inf] = 1.0
    threshold[threshold == inf] = 1.0

    # Subsample the data to 1000 points for plotting
    if len(fpr) > 1000:
        indices = linspace(0, len(fpr) - 1, 1000, dtype=int)
        fpr = fpr[indices]
        tpr = tpr[indices]
        threshold = threshold[indices]
    return fpr, tpr, threshold


def calculate_precision_recall_curve(
    label_column: str,
    positive_class: str | int,
    df: pd.DataFrame,
    predictions: pd.DataFrame,
):
    """Calculates the precision-recall curve for a binary classification problem.

    Args:
        label_column (str): The name of the column containing the true labels.
        positive_class (str | int): The class label for the positive class.
        df (pd.DataFrame): The DataFrame containing the test data.
        predictions (pd.DataFrame): The DataFrame containing the test predictions.
    Returns:
        tuple: A tuple containing the precision, recall, and thresholds.
    """
    y_true_numerical = df[label_column].apply(
        lambda x: 1 if x == positive_class else 0
    )
    precision, recall, thresholds = precision_recall_curve(
        y_true_numerical, predictions[positive_class]
    )

    # Replace inf with 1.0 for plotting
    precision[precision == inf] = 1.0
    recall[recall == inf] = 1.0
    thresholds[thresholds == inf] = 1.0

    # Subsample the data to 1000 points for plotting
    if len(precision) > 1000:
        indices = linspace(0, len(precision) - 1, 1000, dtype=int)
        precision = precision[indices]
        recall = recall[indices]
        thresholds = thresholds[indices]

    return precision, recall, thresholds


def calculate_confusion_matrix(
    label_column: str,
    positive_class: str | int,
    df: pd.DataFrame,
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    """Calculates the confusion matrix for a binary classification problem.

    Args:
        label_column (str): The name of the column containing the true labels.
        positive_class (str | int): The class label for the positive class.
        df (pd.DataFrame): The DataFrame containing the test data.
        predictions (pd.DataFrame): The DataFrame containing the test predictions.
    Returns:
        dict: A dictionary representing the confusion matrix in the format expected by Vertex AI.
    """
    y_true_numerical = df[label_column].apply(
        lambda x: 1 if x == positive_class else 0
    )
    y_pred_numerical = predictions[positive_class].apply(
        lambda x: 1 if x >= 0.5 else 0
    )

    cm = confusion_matrix(y_true_numerical, y_pred_numerical)

    # Convert numpy types to native Python types for JSON serialization
    rows = [[int(c) for c in row] for row in cm]

    annotation_specs = [
        {"displayName": f"not {positive_class}"},
        {"displayName": str(positive_class)},
    ]

    return {
        "annotationSpecs": annotation_specs,
        "rows": rows,
    }


def _generate_threshold_set(
    roc_thresholds: list[float], pr_thresholds: list[float]
) -> list[float]:
    """Generate a comprehensive set of thresholds for evaluation.

    Combines standard Vertex AI thresholds with curve-derived thresholds,
    ensuring 0.5 is included and limiting total count for performance.

    Args:
        roc_thresholds: Thresholds from ROC curve calculation.
        pr_thresholds: Thresholds from Precision-Recall curve calculation.

    Returns:
        Sorted list of thresholds in descending order.
    """
    # Standard thresholds as per Vertex AI specification
    # 0.00, 0.05, 0.10, ..., 0.95, 0.96, 0.97, 0.98, 0.99
    standard_thresholds = [i / 100.0 for i in range(0, 100, 5)]
    standard_thresholds.extend([0.96, 0.97, 0.98, 0.99])

    # Include curve thresholds but limit their number
    curve_thresholds = sorted(set(roc_thresholds + pr_thresholds), reverse=True)
    if len(curve_thresholds) > 50:
        # Keep approximately 25 thresholds from curves
        step = len(curve_thresholds) // 25
        curve_thresholds = curve_thresholds[::step]

    # Combine, deduplicate, and sort
    all_thresholds = sorted(
        set(standard_thresholds + curve_thresholds), reverse=True
    )

    # Limit to reasonable size (Vertex AI typically uses ~100 thresholds max)
    return all_thresholds[:100] if len(all_thresholds) > 100 else all_thresholds


def _calculate_confusion_matrix_components(
    y_true: pd.Series, y_pred: pd.Series
) -> tuple[int, int, int, int]:
    """Calculate confusion matrix components (TP, FP, TN, FN).

    Args:
        y_true: True binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).

    Returns:
        Tuple of (true_positives, false_positives, true_negatives, false_negatives).
    """
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, fp, tn, fn


def _calculate_classification_metrics(
    tp: int, fp: int, tn: int, fn: int
) -> tuple[float, float, float, float]:
    """Calculate classification metrics from confusion matrix components.

    Args:
        tp: True positives count.
        fp: False positives count.
        tn: True negatives count.
        fn: False negatives count.

    Returns:
        Tuple of (recall, precision, false_positive_rate, f1_score).
    """
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1_score = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return recall, precision, fpr, f1_score


def create_vertex_ai_eval(
    label_column: str,
    positive_class: str | int,
    df: pd.DataFrame,
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    """Create evaluation metrics in Vertex AI format.

    Generates comprehensive evaluation metrics including AUC scores, confidence
    metrics across multiple thresholds, and confusion matrices according to
    the Vertex AI classification metrics schema.

    Args:
        label_column: Column name containing true labels.
        positive_class: Label value representing the positive class.
        df: DataFrame containing test data with true labels.
        predictions: DataFrame containing prediction probabilities for each class.

    Returns:
        Dictionary containing evaluation results with:
        - auPrc: Area Under Precision-Recall Curve
        - auRoc: Area Under ROC Curve
        - confidenceMetrics: List of metrics at different thresholds
        - confusionMatrix: Overall confusion matrix at 0.5 threshold
    """
    # Convert labels to binary numerical format
    y_true = df[label_column].apply(lambda x: 1 if x == positive_class else 0)
    y_scores = predictions[positive_class]

    # Calculate overall AUC metrics
    au_roc = float(roc_auc_score(y_true, y_scores))
    au_prc = float(average_precision_score(y_true, y_scores))

    # Get thresholds from curve calculations
    _, _, roc_thresholds = calculate_roc_curve(
        label_column, positive_class, df, predictions
    )
    _, _, pr_thresholds = calculate_precision_recall_curve(
        label_column, positive_class, df, predictions
    )

    # Generate comprehensive threshold set
    thresholds = _generate_threshold_set(
        list(roc_thresholds), list(pr_thresholds)
    )

    # Calculate confusion matrix at standard 0.5 threshold for top-level metric
    confusion_matrix_result = calculate_confusion_matrix(
        label_column, positive_class, df, predictions
    )

    # Generate confidence metrics for each threshold
    confidence_metrics = []
    for threshold in thresholds:
        # Get binary predictions at this threshold
        y_pred = (y_scores >= threshold).astype(int)

        # Calculate confusion matrix components
        tp, fp, tn, fn = _calculate_confusion_matrix_components(y_true, y_pred)

        # Calculate derived metrics
        recall, precision, fpr, f1_score = _calculate_classification_metrics(
            tp, fp, tn, fn
        )

        # Build confidence metric entry
        metric = {
            "confidenceThreshold": float(threshold),
            "recall": recall,
            "precision": precision,
            "falsePositiveRate": fpr,
            "f1Score": f1_score,
            "truePositiveCount": tp,
            "falsePositiveCount": fp,
            "trueNegativeCount": tn,
            "falseNegativeCount": fn,
        }

        # Include confusion matrix for 0.5 threshold
        if abs(threshold - 0.5) < 1e-6:
            metric["confusionMatrix"] = confusion_matrix_result

        confidence_metrics.append(metric)

    return {
        "auPrc": au_prc,
        "auRoc": au_roc,
        "confidenceMetrics": confidence_metrics,
        "confusionMatrix": confusion_matrix_result,
    }
