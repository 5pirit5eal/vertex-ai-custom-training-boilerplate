"""Utility functions for the prediction server."""

import logging
import os
import random
import threading
from typing import Any

import pandas as pd
from autogluon.tabular import TabularPredictor
from google.cloud import storage

PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")
GCS_URI_PREFIX = "gs://"
LOCAL_MODEL_DIR = "/tmp/model/"


logger = logging.getLogger("app.utils")


def download_gcs_dir_to_local(gcs_dir: str, local_dir: str) -> None:
    """Downloads files in a GCS directory to a local directory.

    For example:
      download_gcs_dir_to_local(gs://bucket/foo, /tmp/bar)
      gs://bucket/foo/a -> /tmp/bar/a
      gs://bucket/foo/b/c -> /tmp/bar/b/c

    Args:
      gcs_dir: A string of directory path on GCS.
      local_dir: A string of local directory path.
    """
    if not gcs_dir.startswith("gs://"):
        raise ValueError(f"{gcs_dir} is not a GCS path starting with gs://.")
    bucket_name = gcs_dir.split("/")[2]
    prefix = gcs_dir[len("gs://" + bucket_name) :].strip("/") + "/"
    client = storage.Client(project=PROJECT_ID)
    blobs = client.list_blobs(bucket_name, prefix=prefix)
    for blob in blobs:
        if blob.name[-1] == "/":
            continue
        file_path = blob.name[len(prefix) :].strip("/")
        local_file_path = os.path.join(local_dir, file_path)
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        logger.info(
            "Downloading file from %s to %s",
            file_path,
            local_file_path,
        )
        blob.download_to_filename(local_file_path)


def parse_instances_to_dataframe(
    instances: list[dict[str, Any] | list[Any]],
    predictor: TabularPredictor,
) -> tuple[pd.DataFrame, bool]:
    """Parses request instances and converts them to a DataFrame for inference.

    This function handles both objects (dict) and arrays (list),
    and ensures the resulting DataFrame has the correct feature columns based on the
    predictor's feature metadata.

    If array is provided, the instance values are expected to be in order of the predictor's feature metadata.

    Args:
        instances: Either a list of objects (dict) or arrays (list)
        predictor: The AutoGluon TabularPredictor with loaded model

    Returns:
        A tuple containing:
        - pd.DataFrame: A DataFrame with the correct feature columns for inference.
        - bool: True if the input was a list of lists, False otherwise.

    Raises:
        ValueError: If instances format is invalid or required features are missing
    """
    # Get the expected feature names from the predictor
    feature_metadata = predictor.feature_metadata_in.to_dict()
    expected_features = list(feature_metadata.keys())

    if not instances:
        return pd.DataFrame(columns=expected_features), False

    first_instance = instances[0]
    is_list = isinstance(first_instance, list)

    if isinstance(first_instance, dict):
        df = pd.DataFrame(instances)
        missing_features = set(expected_features) - set(df.columns)
        if missing_features:
            raise ValueError(
                f"Missing required features: {list(missing_features)}"
            )
        return df[expected_features], is_list

    elif isinstance(first_instance, list):
        num_expected_features = len(expected_features)
        for i, instance in enumerate(instances):
            if len(instance) != num_expected_features:
                raise ValueError(
                    f"Instance {i} has {len(instance)} values, but {num_expected_features} are expected."
                )
        return pd.DataFrame(instances, columns=expected_features), is_list

    raise ValueError(
        "Invalid instances format. Provide a list of JSON objects (dict) or a list of arrays (list)."
    )


def load_model() -> TabularPredictor:
    """Loads the model from the path specified by the AIP_STORAGE_URI.

    This function includes logic to handle multiple workers trying to download the model
    at the same time.
    """
    # Wait the thread for a random few seconds to avoid race conditions
    threading.Event().wait(random.randint(0, 5))

    model_dir = os.getenv("AIP_STORAGE_URI", "/model/")
    logger.info("Model directory %s passed by user", model_dir)

    if model_dir.startswith(GCS_URI_PREFIX):
        gcs_path = model_dir[len(GCS_URI_PREFIX) :]
        local_model_dir = os.path.join(LOCAL_MODEL_DIR, gcs_path)

        if os.path.exists(local_model_dir):
            # Other workers might be downloading the model, wait until it is available
            # version.txt is the last file to be downloaded, so we wait for it
            while not os.path.exists(
                os.path.join(local_model_dir, "version.txt")
            ):
                logger.info("Waiting until Model is finished downloading...")
                threading.Event().wait(15)

            # Ensure the version.txt file is fully loaded before proceeding
            threading.Event().wait(5)
        else:
            logger.info(
                "Downloading model from %s to %s",
                model_dir,
                local_model_dir,
            )
            download_gcs_dir_to_local(model_dir, local_model_dir)
            logger.info("Finished downloading model to %s", local_model_dir)

        return TabularPredictor.load(local_model_dir)

    else:
        logger.info("Model directory %s is local", model_dir)
        if not os.path.exists(model_dir):
            raise FileNotFoundError(
                f"Model directory {model_dir} does not exist."
            )
        return TabularPredictor.load(model_dir)
