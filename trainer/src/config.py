# Task type can be either 'classification', 'regression', or 'custom'.
# This is based on the target feature in the dataset.
import os
import msgspec
from typing import Literal
import click


class Hyperparameters(msgspec.Struct):
    """Configuration of hyperparameters for the training job."""

    # TODO: Hyperparameters for the training job
    tbd: str | None = None


class Config(msgspec.Struct):
    """Holds the configuration for the training job."""

    # Vertex AI custom training variables
    project_id: str
    region: str
    data_format: Literal["csv", "jsonl", "bigquery"]
    train_data_uri: str
    val_data_uri: str
    test_data_uri: str
    model_import_uri: str | None
    model_export_uri: str | None
    checkpoint_uri: str | None
    tensorboard_log_uri: str | None

    # Model training variables
    label: str
    task_type: Literal["binary", "multiclass", "regression", "quantile"]
    eval_metric: str
    preset: str
    # hyperparameters: dict

    def __post_init__(self) -> None:
        """Validates the metadata for correctness."""
        if self.data_format == "bigquery":
            # Check that the data URIS are in the correct format for BigQuery
            for uri in [
                self.train_data_uri,
                self.val_data_uri,
                self.test_data_uri,
            ]:
                if not uri.startswith("bq://"):
                    raise ValueError(f"Invalid data URI for BigQuery: {uri}")
        else:
            # Check that the data URIS are in the correct format for CSV or JSONL
            for uri in [
                self.train_data_uri,
                self.val_data_uri,
                self.test_data_uri,
            ]:
                if not uri.startswith("gs://"):
                    raise ValueError(
                        f"Invalid data URI for {self.data_format}: {uri}"
                    )

    @classmethod
    @click.command()
    @click.option(
        "--project-id",
        help="Google Cloud project ID",
        default=os.getenv("CLOUD_ML_PROJECT_ID"),
    )
    @click.option(
        "--region",
        help="Google Cloud region",
        default=os.getenv("REGION"),
    )
    @click.option(
        "--data-format",
        help="Data format (csv, jsonl, bigquery)",
        type=click.Choice(["csv", "jsonl", "bigquery"]),
        default=os.getenv("AIP_DATA_FORMAT"),
    )
    @click.option(
        "--train-data-uri",
        help="Training data URI",
        default=os.getenv("AIP_TRAINING_DATA_URI"),
    )
    @click.option(
        "--val-data-uri",
        help="Validation data URI",
        default=os.getenv("AIP_VALIDATION_DATA_URI"),
    )
    @click.option(
        "--test-data-uri",
        help="Testing data URI",
        default=os.getenv("AIP_TESTING_DATA_URI"),
    )
    @click.option(
        "--model-import-uri",
        help="Model import URI",
        default=os.getenv("AIP_MODEL_DIR"),
    )
    @click.option(
        "--model-export-uri",
        help="Model export URI",
        default=os.getenv("AIP_MODEL_DIR"),
    )
    @click.option(
        "--checkpoint-uri",
        help="Checkpoint URI",
        default=os.getenv("AIP_CHECKPOINT_DIR"),
    )
    @click.option(
        "--tensorboard-log-uri",
        help="TensorBoard log URI",
        default=os.getenv("AIP_TENSORBOARD_LOG_DIR"),
    )
    def load(cls, **kwargs) -> "Config":
        """Loads the metadata from environment variables including preconfigured vertex ai custom training variables."""
        return msgspec.convert(kwargs, cls, from_attributes=True)
