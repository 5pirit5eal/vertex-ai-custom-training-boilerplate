"""Tests for the trainer.data module."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from trainer.config import Config
from trainer.data import gcs_path, load_split_df


def test_gcs_path():
    """Tests the gcs_path function."""
    assert gcs_path("gs://bucket/path") == "/gcs/bucket/path"
    assert (
        gcs_path("gs://bucket/path", "file.csv") == "/gcs/bucket/path/file.csv"
    )
    with pytest.raises(ValueError, match="Invalid GCS URI: not-a-gcs-uri"):
        gcs_path("not-a-gcs-uri")


@patch("os.makedirs")
@patch("trainer.data.get_bigquery_client")
def test_load_split_df_bigquery(mock_get_bq_client, mock_makedirs):
    """Tests loading data from BigQuery."""
    mock_client = MagicMock()
    mock_get_bq_client.return_value = mock_client
    mock_client.query.return_value.to_dataframe.return_value = pd.DataFrame(
        {"col1": [1, 2, 3]}
    )

    config = Config(
        project_id="test-project",
        region="us-central1",
        log_level="INFO",
        data_format="bigquery",
        train_data_uri="bq://project.dataset.table",
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

    df = load_split_df(config, "train")
    assert not df.empty
    mock_client.query.assert_called_once_with(
        "SELECT * FROM `bq://project.dataset.table` WHERE split = 'train'"
    )


@patch("os.makedirs")
@patch("trainer.data.pd.read_csv")
@patch("trainer.data.gcs_path", return_value="/gcs/test-bucket/train.csv")
def test_load_split_df_csv(mock_gcs_path, mock_read_csv, mock_makedirs):
    """Tests loading data from a CSV file in GCS."""
    mock_read_csv.return_value = pd.DataFrame({"col1": [1, 2, 3]})
    config = Config(
        project_id="test-project",
        region="us-central1",
        log_level="INFO",
        data_format="csv",
        train_data_uri="gs://test-bucket/train.csv",
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

    df = load_split_df(config, "train")
    assert not df.empty
    mock_read_csv.assert_called_once_with("/gcs/test-bucket/train.csv")
