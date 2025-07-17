import os
from typing import Any

import pandas as pd
from google.cloud import storage
from autogluon.tabular import TabularPredictor

PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")


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
        print("Downloading", file_path, "to", local_file_path)
        blob.download_to_filename(local_file_path)


def parse_instances_to_dataframe(
    instances: list[dict[str, Any]] | list[list[Any]],
    predictor: TabularPredictor,
) -> pd.DataFrame:
    """Parses request instances and converts them to a DataFrame for inference.

    This function handles both objects (dict) and arrays (list),
    and ensures the resulting DataFrame has the correct feature columns based on the
    predictor's feature metadata.

    If array is provided, the instance values are expected to be in order of the predictor's feature metadata.

    Args:
        instances: Either a list of objects (dict) or arrays (list)
        predictor: The AutoGluon TabularPredictor with loaded model

    Returns:
        pd.DataFrame: A DataFrame with the correct feature columns for inference

    Raises:
        ValueError: If instances format is invalid or required features are missing
    """
    # Get the expected feature names from the predictor
    feature_metadata = predictor.feature_metadata_in.to_dict()
    expected_features = list(feature_metadata.keys())

    if not instances:
        return pd.DataFrame(columns=expected_features)

    first_instance = instances[0]

    if isinstance(first_instance, dict):
        df = pd.DataFrame(instances)
        missing_features = set(expected_features) - set(df.columns)
        if missing_features:
            raise ValueError(
                f"Missing required features: {list(missing_features)}"
            )
        return df[expected_features]

    if isinstance(first_instance, list):
        num_expected_features = len(expected_features)
        for i, instance in enumerate(instances):
            if len(instance) != num_expected_features:
                raise ValueError(
                    f"Instance {i} has {len(instance)} values, but {num_expected_features} are expected."
                )
        return pd.DataFrame(instances, columns=expected_features)

    raise ValueError(
        "Invalid instances format. Provide a list of JSON objects (dict) or a list of arrays (list)."
    )
