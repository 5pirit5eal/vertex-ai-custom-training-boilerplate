import os

from google.cloud import bigquery

from config import Config


def get_bigquery_client(metadata: Config) -> bigquery.Client:
    """Creates a BigQuery client using the project ID from the metadata."""
    return bigquery.Client(
        project=metadata.project_id, location=metadata.region
    )


def convert_gs_to_gcs(uri: str) -> str:
    """Converts a Google Cloud Storage URI to a Google Cloud Storage FUSE path."""
    if uri.startswith("gs://"):
        return "/gcs/" + uri[5:]
    else:
        raise ValueError(f"Invalid GCS URI: {uri}")
