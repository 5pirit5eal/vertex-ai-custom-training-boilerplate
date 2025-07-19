"""Module containing utility functions for working with Vertex AI products.

The functions create schemas for model instance and prediction, and evaluation metrics for easier upload to Vertex AI.
"""

import logging
import os
from typing import Any

import pandas as pd
import yaml
from autogluon.tabular import TabularPredictor
from numpy import inf, linspace
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from trainer.config import Config
from trainer.data import gcs_path


def _convert_feature_map_to_input_schema(
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


def _convert_class_labels_to_pred_schema(
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

        feature_schema = _convert_feature_map_to_input_schema(
            feature_metadata_dict
        )
        label_schema = _convert_class_labels_to_pred_schema(classes)

        parameters_schema = {
            "type": "object",
            "properties": {
                "as_object": {
                    "type": "boolean",
                    "description": "If true, the prediction response will be a JSON object. If false, it will be a JSON array. If not provided, the format will be inferred from the input.",
                }
            },
        }

        # Write the schema to a YAML file
        for schema, filename in [
            (feature_schema, "instance_schema.yaml"),
            (label_schema, "prediction_schema.yaml"),
            (parameters_schema, "parameters_schema.yaml"),
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
    labels: list[str | int] | None = None,
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

    cm = confusion_matrix(y_true_numerical, y_pred_numerical, labels=labels)

    # Convert numpy types to native Python types for JSON serialization
    rows = [{"row": [int(c) for c in row]} for row in cm]

    if labels is None:
        annotation_specs = [
            {"displayName": f"not {positive_class}"},
            {"displayName": str(positive_class)},
        ]
    else:
        annotation_specs = [{"displayName": str(label)} for label in labels]

    return {
        "annotationSpecs": annotation_specs,
        "rows": rows,
    }


def _generate_threshold_set(y_scores: pd.Series) -> list[float]:
    """Generate a comprehensive set of thresholds based on the confidence range given.

    This function ensures 0.5 is included and limiting total count for performance.

    Args:
        y_scores (pd.Series): The series of prediction scores.

    Returns:
        Sorted list of thresholds in descending order.
    """
    # Get unique scores and sort them in descending order
    unique_scores = sorted(y_scores.unique(), reverse=True)

    # Ensure 0.5 is included
    if 0.5 not in unique_scores:
        unique_scores.append(0.5)
        unique_scores.sort(reverse=True)

    # Limit the number of thresholds to a maximum of 1000 for performance
    max_thresholds = 1000
    if len(unique_scores) > max_thresholds:
        step = len(unique_scores) // max_thresholds
        thresholds = unique_scores[::step]
    else:
        thresholds = unique_scores

    return thresholds


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
    labels: list[str | int] | None = None,
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
        labels: Optional list of class labels for confusion matrix annotation.

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

    # Generate comprehensive threshold set
    thresholds = _generate_threshold_set(y_scores)

    # Calculate confusion matrix at standard 0.5 threshold for top-level metric
    confusion_matrix_result = calculate_confusion_matrix(
        label_column, positive_class, df, predictions, labels=labels
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


def create_multiclass_vertex_ai_eval(
    label_column: str,
    df: pd.DataFrame,
    predictions: pd.DataFrame,
    labels: list[str | int],
) -> dict[str, Any]:
    """Create evaluation metrics for multiclass classification in Vertex AI format.

    Args:
        label_column (str): The name of the column containing the true labels.
        df (pd.DataFrame): The DataFrame containing the test data.
        predictions (pd.DataFrame): The DataFrame containing the test predictions.

    Returns:
        dict: A dictionary containing the evaluation metrics.
    """
    # Calculate confusion matrix
    y_true = df[label_column]
    y_pred = predictions.idxmax(axis=1)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    annotation_specs = [{"displayName": str(label)} for label in labels]

    return {
        "confusionMatrix": {
            "annotationSpecs": annotation_specs,
            "rows": [{"row": [int(c) for c in row]} for row in cm],
        }
    }
