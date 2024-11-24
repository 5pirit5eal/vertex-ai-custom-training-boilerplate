import json
import logging
import os
import random
import string

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import xgboost as xgb
from google.cloud import storage  # type: ignore[import-untyped]
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]


def preprocess_function(
    data: pd.DataFrame, labels: pd.DataFrame | pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame | pd.Series]:
    """Preprocesses the data and labels.

    Args:
        data (pd.DataFrame): The data to preprocess.
        labels (pd.DataFrame): The labels to preprocess.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame | pd.Series]: The preprocessed data and labels.
    """
    # NOTE: Implement your data cleaning and non-model preprocessing here
    gender_encoded = data["gender"].str.get_dummies()
    smoking_history_categories = ["never", "former", "not current", "current"]

    # replace all other values in smoking history with None
    data["smoking_history"] = (
        data["smoking_history"].where(
            data["smoking_history"].isin(smoking_history_categories), np.nan
        )
    ).replace(smoking_history_categories, [0, 1, 2, 3])

    data = data.drop(columns=["gender"])
    data = pd.concat([data, gender_encoded], axis=1)
    return data, labels


def load_data(
    gs_dir: str, model_name: str, model_id: str
) -> tuple[
    list[pd.DataFrame | pd.Series],
    list[pd.DataFrame | pd.Series],
    list[pd.DataFrame | pd.Series],
]:
    """Loads the data from Google Cloud Storage FUSE or local file system and
    returns the train and test datasets.

    Args:
        gs_dir (str): The directory to load data from and save the split to.

    Returns:
        tuple[tuple[pd.DataFrame, pd.DataFrame | pd.Series],
        tuple[pd.DataFrame, pd.DataFrame | pd.Series],
        tuple[pd.DataFrame, pd.DataFrame | pd.Series]:
            train, validation and test data tuples
    """
    logging.info("Creating training data and labels")

    df = pd.read_csv(os.path.join(gs_dir, "data", "diabetes_prediction_dataset.csv"))

    df_data = df.drop(columns=["diabetes"])
    df_labels = df["diabetes"]

    df_data_processed, df_labels_processed = preprocess_function(df_data, df_labels)

    train_data, test_data, train_labels, test_labels = train_test_split(
        df_data_processed,
        df_labels_processed,
        test_size=0.2,
        random_state=7,
    )

    # split test data into test and validation
    test_data, validation_data, test_labels, validation_labels = train_test_split(
        test_data,
        test_labels,
        test_size=0.5,
        random_state=42,
    )
    # Concat labels and data
    train_df = pd.concat([train_data, train_labels], axis=1)
    val_df = pd.concat([validation_data, validation_labels], axis=1)
    test_df = pd.concat([test_data, test_labels], axis=1)

    # Create the model directory
    model_dir = os.path.join(model_name, model_id)
    os.makedirs(model_dir, exist_ok=True)

    # save the preprocessed data and labels
    train_df.to_csv(os.path.join(model_dir, "train_data.csv"), index=False)
    val_df.to_csv(os.path.join(model_dir, "val_data.csv"), index=False)
    test_df.to_csv(os.path.join(model_dir, "test_data.csv"), index=False)

    if gs_dir.startswith("gs://"):
        storage_client = storage.Client(project=os.getenv("PROJECT_ID"))

        # Upload the data to google cloud
        for file in ["train_data.csv", "val_data.csv", "test_data.csv"]:
            bucket = storage_client.bucket(gs_dir.split("/")[2])
            blob = bucket.blob(os.path.join(model_dir, file))
            blob.upload_from_filename(os.path.join(model_dir, file))

    return (
        [train_data, train_labels.to_frame()],
        [validation_data, validation_labels.to_frame()],
        [test_data, test_labels.to_frame()],
    )


def generate_random_id(length=8):
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))


def save_model_and_results(
    model: xgb.XGBClassifier,
    accuracy: float,
    gs_dir: str,
    model_name: str,
    model_id: str,
):
    """Saves the model to Google Cloud Storage or local file system

    Args:
        model (XGBClassifier): The model to save.
        accuracy (float): The accuracy of the model.
        gs_dir (str): The directory to save the model to.
        model_name (str): The name of the model.
        model_id (str): The ID of the model.
    """
    eval_metrics = model.evals_result()
    eval_metrics["test"] = {"accuracy_score": accuracy}

    if gs_dir.startswith("gs://"):
        storage_client = storage.Client(project=os.getenv("PROJECT_ID"))
        bucket = storage_client.bucket(gs_dir.split("/")[2])
        blob = bucket.blob(os.path.join(model_name, model_id, "model.json"))
        logging.info("Saving model artifacts to %s", blob.public_url)
        model.save_model("model.json")
        blob.upload_from_filename("model.json")
        logging.info(
            "Saving metrics to %s/%s/%s/metrics.json",
            gs_dir,
            model_name,
            model_id,
            extra={"json_fields": eval_metrics},
        )
        blob = bucket.blob(os.path.join(model_name, model_id, "metrics.json"))
        blob.upload_from_string(json.dumps(eval_metrics))
    else:
        logging.info("Saving model artifacts and eval metrics locally")
        model.save_model(f"~/{model_name}/{model_id}/model.bst")
        with open(f"~/{model_name}/{model_id}/metrics.json", "w") as f:
            f.write(json.dumps(eval_metrics))
