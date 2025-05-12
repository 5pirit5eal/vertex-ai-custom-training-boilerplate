import json
import logging
from fnmatch import fnmatch
from typing import Literal

import pandas as pd
from google.cloud import bigquery, storage

from config import Config


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
        except Exception as e:
            df = load_wildcard_csv(data_uri)
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
    blobs: list[storage.Blob] = bucket.list_blobs(prefix=prefix)

    # Load each CSV file into a DataFrame and concatenate them
    return pd.concat(
        [
            pd.read_csv(convert_gs_to_gcs(blob.name))
            for blob in blobs
            if fnmatch(blob.name, uri)
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
    filenname: str,
) -> None:
    """Writes a DataFrame to a GCS URI.

    Args:
        config (Config): The configuration object containing the data URI.
        df (pd.DataFrame): The DataFrame to write.
        filenname (str): The filename to write to.
    """
    # Convert the GCS URI to a GCS FUSE path
    uri = convert_gs_to_gcs(config.model_export_uri)
    uri = uri + "/" + filenname

    # Write the DataFrame to a CSV file in GCS
    df.to_csv(uri, index=False)


def write_json(
    config: Config,
    data: dict,
    filenname: str,
) -> None:
    """Writes a dict as JSON to a GCS URI.
    Args:
        config (Config): The configuration object containing the data URI.
        data (dict): The dict to write.
        filenname (str): The filename to write to.
    """
    # Convert the GCS URI to a GCS FUSE path
    uri = convert_gs_to_gcs(config.model_export_uri)
    uri = uri + "/" + filenname

    # Write the dict to a JSON file in GCS
    with open(uri, "w") as f:
        json.dump(data, f)
