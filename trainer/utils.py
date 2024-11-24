import json
import os
import logging

import pandas as pd
from xgboost import XGBModel

from sklearn.model_selection import train_test_split

from trainer import metadata


def preprocess_function(
    data: pd.DataFrame, labels: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Preprocesses the data and labels.

    Args:
        data (pd.DataFrame): The data to preprocess.
        labels (pd.DataFrame): The labels to preprocess.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The preprocessed data and labels.
    """
    # NOTE: Implement your data cleaning and non-model preprocessing here
    return data, labels


def load_data(
    gs_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame]]:
    """Loads the data from Google Cloud Storage FUSE or local file system and
    returns the train and test datasets.

    Args:
        gs_dir (str): The directory to save the split to.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame]]:
            train_data and labels as dataframes, and eval_tuple
    """
    logging.info("Creating training data and labels")

    df = pd.read_csv(metadata.DATA)

    df_data = df.drop(columns=["diabetes"])
    df_labels = df["diabetes"]

    df_data_processed, df_labels_processed = preprocess_function(df_data, df_labels)

    train_data, test_data, train_labels, test_labels = train_test_split(
        df_data_processed,
        df_labels_processed,
        test_size=0.2,
        random_state=7,
    )

    # Concat labels and data
    train_df = pd.concat([train_data, train_labels], axis=1)
    test_df = pd.concat([test_data, test_labels], axis=1)

    # save the preprocessed data and labels
    train_df.to_csv(os.path.join(gs_dir, "train_data.csv"), index=False)
    test_df.to_csv(os.path.join(gs_dir, "test_data.csv"), index=False)

    return train_data, train_labels, (test_data, test_labels)


def convert_gs_to_gcs(gs_dir: str):
    gs_prefix = "gs://"
    gcs_prefix = "/gcs/"
    if gs_dir.startswith(gs_prefix):
        gs_dir = gs_dir.replace(gs_prefix, gcs_prefix)
        if not os.path.isdir(os.path.split(gs_dir)[0]):
            os.makedirs(os.path.split(gs_dir)[0])

    return gs_dir


def save_model(
    model: XGBModel,
    gs_dir: str,
    model_name: str,
):
    """Saves the model to Google Cloud Storage or local file system

    Args:
        model (xgb.XGBClassifier): The model to save.
        gs_dir (str): The directory to save the model to.
        model_name (str): The name of the model.
    """
    if gs_dir.startswith("/gcs/"):
        gcs_model_path = os.path.join(gs_dir, model_name, "model.bst")
        logging.info("Saving model artifacts to %s", gcs_model_path)
        model.save_model(gcs_model_path)
        logging.info(
            "Saving metrics to %s/%s/metrics.json",
            gs_dir,
            model_name,
            extra={"json_fields": model.eval_results()},
        )
        gcs_metrics_path = os.path.join(gs_dir, model_name, "metrics.json")
        with open(gcs_metrics_path, "w") as f:
            f.write(json.dumps(model.eval_results()))
    else:
        model.save_model(f"~/{model_name}/model.bst")
