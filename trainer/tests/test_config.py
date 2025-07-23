"""Tests for the trainer.config module."""

import os
from unittest.mock import patch

import pytest

from trainer.config import Config, load_config


@patch("os.makedirs")
def test_config_post_init_valid(mock_makedirs):
    """Tests that the Config object is initialized correctly with valid data."""
    config = Config(
        project_id="test-project",
        region="us-central1",
        log_level="INFO",
        data_format="csv",
        train_data_uri="gs://test-bucket/train.csv",
        val_data_uri="gs://test-bucket/val.csv",
        test_data_uri="gs://test-bucket/test.csv",
        model_import_uri=None,
        model_export_uri="gs://test-bucket/model",
        checkpoint_uri="gs://test-bucket/checkpoints",
        tensorboard_log_uri="gs://test-bucket/logs",
        time_limit=None,
        label="target",
        task_type=None,
        eval_metric=None,
        presets=["best_quality"],
    )
    assert config.project_id == "test-project"


@patch("os.makedirs")
def test_config_post_init_invalid_gcs_uri(mock_makedirs):
    """Tests that a ValueError is raised when an invalid GCS URI is provided."""
    with pytest.raises(
        ValueError, match="Invalid data URI for csv: invalid-uri"
    ):
        Config(
            project_id="test-project",
            region="us-central1",
            log_level="INFO",
            data_format="csv",
            train_data_uri="invalid-uri",
            val_data_uri=None,
            test_data_uri=None,
            model_import_uri=None,
            model_export_uri="gs://test-bucket/model",
            checkpoint_uri="gs://test-bucket/checkpoints",
            tensorboard_log_uri="gs://test-bucket/logs",
            time_limit=None,
            label="target",
            task_type=None,
            eval_metric=None,
            presets=["best_quality"],
        )


@patch("os.makedirs")
def test_config_post_init_invalid_bq_uri(mock_makedirs):
    """Tests that a ValueError is raised when an invalid BigQuery URI is provided."""
    with pytest.raises(
        ValueError, match="Invalid data URI for BigQuery: invalid-uri"
    ):
        Config(
            project_id="test-project",
            region="us-central1",
            log_level="INFO",
            data_format="bigquery",
            train_data_uri="invalid-uri",
            val_data_uri=None,
            test_data_uri=None,
            model_import_uri=None,
            model_export_uri="gs://test-bucket/model",
            checkpoint_uri="gs://test-bucket/checkpoints",
            tensorboard_log_uri="gs://test-bucket/logs",
            time_limit=None,
            label="target",
            task_type=None,
            eval_metric=None,
            presets=["best_quality"],
        )


@patch("os.makedirs")
@patch.dict(
    os.environ,
    {
        "CLOUD_ML_PROJECT_ID": "test-project-env",
        "CLOUD_ML_REGION": "us-central1-env",
        "AIP_DATA_FORMAT": "csv",
        "AIP_TRAINING_DATA_URI": "gs://test-bucket/train-env.csv",
        "AIP_MODEL_DIR": "gs://test-bucket/model-env",
        "AIP_CHECKPOINT_DIR": "gs://test-bucket/checkpoints-env",
        "AIP_TENSORBOARD_LOG_DIR": "gs://test-bucket/logs-env",
        "AIP_LABEL_COLUMN": "target-env",
    },
    clear=True,
)
def test_load_config_from_env(mock_makedirs):
    """Tests that the config is loaded correctly from environment variables."""
    config = load_config.main(
        [
            "--project-id",
            "test-project-env",
            "--region",
            "us-central1-env",
            "--data-format",
            "csv",
            "--train-data-uri",
            "gs://test-bucket/train-env.csv",
            "--model-export-uri",
            "gs://test-bucket/model-env",
            "--checkpoint-uri",
            "gs://test-bucket/checkpoints-env",
            "--tensorboard-log-uri",
            "gs://test-bucket/logs-env",
            "--label",
            "target-env",
        ],
        standalone_mode=False,
    )
    assert config.project_id == "test-project-env"
    assert config.region == "us-central1-env"
    assert config.train_data_uri == "gs://test-bucket/train-env.csv"
    assert config.model_export_uri == "gs://test-bucket/model-env"
    assert config.label == "target-env"
