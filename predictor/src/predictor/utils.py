import fnmatch
from google.cloud import storage
import os

PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")
BUCKET_URI = os.getenv(
    "BUCKET_URI", "gs://fd-pubfox-adesso-tmp-20250505155632-7025"
)


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
