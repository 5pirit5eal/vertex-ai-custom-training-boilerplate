"""This module defines the configuration for the training job."""

import os
from typing import Any, Literal

import click
import msgspec
from autogluon.tabular.configs.hyperparameter_configs import (
    get_hyperparameter_config,
)


class Config(msgspec.Struct):
    """Holds the configuration for the training job."""

    # Vertex AI custom training variables
    project_id: str
    region: str
    log_level: str
    data_format: Literal["csv", "bigquery"]
    train_data_uri: str
    val_data_uri: str | None
    test_data_uri: str | None
    model_import_uri: str | None
    model_export_uri: str
    checkpoint_uri: str
    tensorboard_log_uri: str

    # Model training variables
    time_limit: int | None
    label: str
    task_type: Literal["binary", "multiclass", "regression", "quantile"] | None
    eval_metric: str | None
    presets: str | list[str]
    use_gpu: bool = False
    calculate_importance: bool = False
    hyperparameters: dict[str, Any] | None = None
    multimodal: bool = False
    refit_full: bool = False
    sample_weight: str = "weight"

    # Vertex AI experiment variables
    experiment_name: str | None = None
    experiment_run_name: str | None = None
    resume: bool = False

    def __post_init__(self) -> None:
        """Validates the metadata for correctness."""
        if self.data_format == "bigquery":
            # Check that the data URIS are in the correct format for BigQuery
            for uri in [
                self.train_data_uri,
                self.val_data_uri,
                self.test_data_uri,
            ]:
                if isinstance(uri, str) and not uri.startswith("bq://"):
                    raise ValueError(f"Invalid data URI for BigQuery: {uri}")
        else:
            # Check that the data URIS are in the correct format for CSV or JSONL
            for uri in [
                self.train_data_uri,
                self.val_data_uri,
                self.test_data_uri,
            ]:
                if isinstance(uri, str) and not uri.startswith("gs://"):
                    raise ValueError(
                        f"Invalid data URI for {self.data_format}: {uri}"
                    )

        if self.model_export_uri is None:
            raise ValueError(
                "Model export URI must be provided. Set AIP_MODEL_DIR environment variable or provide it as an arg."
            )

        if self.tensorboard_log_uri is None:
            # If no tensorboard log URI is provided, use the model export URI
            self.tensorboard_log_uri = os.path.join(
                self.model_export_uri, "logs"
            )

        if self.checkpoint_uri is None:
            # If no tensorboard log URI is provided, use the model export URI
            self.tensorboard_log_uri = os.path.join(
                self.model_export_uri, "checkpoints"
            )

        # Create the necessary folders
        for uri in [
            self.model_export_uri,
            self.checkpoint_uri,
            self.tensorboard_log_uri,
        ]:
            if uri is not None:
                uri = uri.strip()
                if uri.startswith("gs://"):
                    os.makedirs("/gcs/" + uri[5:], exist_ok=True)
                else:
                    raise ValueError(f"Invalid GCS URI: {uri}")

        # Check that only one quality preset is used
        if (
            len(
                [
                    preset
                    for preset in self.presets
                    if "quality" in preset.lower()
                ]
            )
            > 1
        ):
            raise ValueError(
                f"Only one quality preset can be used. Found: {self.presets}"
            )

        # Set the hyperparameters if multimodal is True
        if self.multimodal:
            if self.hyperparameters is None:
                self.hyperparameters = {}
            self.hyperparameters.update(get_hyperparameter_config("multimodal"))

        if (
            self.experiment_run_name is not None
            and self.experiment_name is None
        ):
            raise ValueError(
                "Experiment name must be provided if experiment run name is set."
            )

    def to_dict(self) -> dict[str, Any]:
        """Converts the Config object to a dictionary."""
        return {
            key: value
            for key, value in msgspec.to_builtins(self).items()
            if isinstance(value, (str, int, bool, float))
        }


@click.command(name="train")
@click.option(
    "--project-id",
    help="Google Cloud project ID",
    default=os.getenv("CLOUD_ML_PROJECT_ID"),
)
@click.option(
    "--region",
    help="Google Cloud region",
    default=os.getenv("CLOUD_ML_REGION"),
)
@click.option(
    "--log-level",
    help="Logging level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    default=os.getenv("CLOUD_ML_LOG_LEVEL", "DEBUG"),
)
@click.option(
    "--data-format",
    help="Data format (csv, jsonl, bigquery)",
    type=click.Choice(["csv", "bigquery"]),
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
    default=os.getenv("AIP_TEST_DATA_URI"),
)
@click.option(
    "--model-import-uri",
    help="Model import URI",
    default=None,
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
@click.option(
    "--label",
    help="Label column name",
    default=os.getenv("AIP_LABEL_COLUMN"),
)
@click.option(
    "--task-type",
    help="Task type (binary, multiclass, regression, quantile)",
    type=click.Choice(["binary", "multiclass", "regression", "quantile"]),
    default=None,
)
@click.option("--eval-metric", help="Evaluation metric", default=None)
@click.option(
    "--presets",
    help="Preset for the training job",
    default="best_quality",
    type=click.Choice(
        [
            "best_quality",
            "high_quality",
            "good_quality",
            "medium_quality",
            "experimental_quality",
            "interpretable",
            "ignore_text",
        ]
    ),
)
@click.option(
    "--time-limit",
    help="Time limit for the training job in seconds",
    type=int,
    default=None,
)
@click.option(
    "--use-gpu",
    help="Use GPU for training",
    is_flag=True,
    default=False,
)
@click.option(
    "--multimodal",
    help="Use multimodal training",
    is_flag=True,
    default=False,
)
@click.option(
    "--refit-full",
    help="Refit the model on the full training data after initial training, collapsing the ensemble. This will speed up inference but may reduce accuracy.",
    is_flag=True,
    default=False,
)
@click.option(
    "--calculate-importance",
    help="Calculate feature importance",
    is_flag=True,
    default=False,
)
@click.option(
    "--sample-weight",
    help="Weight configuration for the training data. Either a column name, 'auto_weight' "
    "for automatic weighting or 'balanced_weight' for balanced weighting.",
    default="weight",
)
@click.option(
    "--experiment-name",
    help="Experiment name for Vertex AI",
)
@click.option(
    "--experiment-run-name",
    help="Experiment run name for Vertex AI",
)
@click.option(
    "--resume",
    help="Resume the experiment run if it already exists. This will also cause the experiment to not stop the run.",
    is_flag=True,
    default=False,
)
def load_config(**kwargs) -> Config:
    """Loads the metadata from environment variables including preconfigured vertex ai custom training variables."""
    return msgspec.convert(kwargs, Config, strict=False)
