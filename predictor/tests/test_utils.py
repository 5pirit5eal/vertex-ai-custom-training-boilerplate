"""Tests for the utility functions."""

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from autogluon.tabular import TabularPredictor

from predictor.utils import (
    download_gcs_dir_to_local,
    parse_instances_to_dataframe,
)


@pytest.fixture
def mock_predictor() -> MagicMock:
    """Returns a mock TabularPredictor."""
    predictor = MagicMock(spec=TabularPredictor)
    predictor.feature_metadata_in.to_dict.return_value = {
        "feat1": "int",
        "feat2": "float",
    }
    return predictor


def test_parse_instances_to_dataframe_with_dict(
    mock_predictor: MagicMock,
) -> None:
    """Tests parse_instances_to_dataframe with a list of dictionaries."""
    instances: list[dict[str, Any] | list[Any]] = [{"feat1": 1, "feat2": 2.0}]
    df, is_list = parse_instances_to_dataframe(instances, mock_predictor)
    assert isinstance(df, pd.DataFrame)
    assert not is_list
    assert df.to_dict(orient="records") == instances


def test_parse_instances_to_dataframe_with_list(
    mock_predictor: MagicMock,
) -> None:
    """Tests parse_instances_to_dataframe with a list of lists."""
    instances: list[dict[str, Any] | list[Any]] = [[1, 2.0]]
    df, is_list = parse_instances_to_dataframe(instances, mock_predictor)
    assert isinstance(df, pd.DataFrame)
    assert is_list
    assert df.values.tolist() == instances


def test_parse_instances_to_dataframe_with_missing_features(
    mock_predictor: MagicMock,
) -> None:
    """Tests parse_instances_to_dataframe with missing features."""
    instances: list[dict[str, Any] | list[Any]] = [{"feat1": 1}]
    with pytest.raises(ValueError, match="Missing required features"):
        parse_instances_to_dataframe(instances, mock_predictor)


def test_parse_instances_to_dataframe_with_wrong_number_of_features(
    mock_predictor: MagicMock,
) -> None:
    """Tests parse_instances_to_dataframe with the wrong number of features."""
    instances: list[dict[str, Any] | list[Any]] = [[1]]
    with pytest.raises(
        ValueError, match="Instance 0 has 1 values, but 2 are expected."
    ):
        parse_instances_to_dataframe(instances, mock_predictor)


def test_parse_instances_to_dataframe_with_empty_list(
    mock_predictor: MagicMock,
) -> None:
    """Tests parse_instances_to_dataframe with an empty list."""
    df, is_list = parse_instances_to_dataframe([], mock_predictor)
    assert isinstance(df, pd.DataFrame)
    assert not is_list
    assert df.empty


def test_parse_instances_to_dataframe_with_invalid_format(
    mock_predictor: MagicMock,
) -> None:
    """Tests parse_instances_to_dataframe with an invalid format."""
    with pytest.raises(ValueError, match="Invalid instances format"):
        parse_instances_to_dataframe("invalid", mock_predictor)  # type: ignore


@patch("predictor.utils.storage.Client")
def test_download_gcs_dir_to_local(mock_storage_client: MagicMock) -> None:
    """Tests the download_gcs_dir_to_local function."""
    mock_blob = MagicMock()
    mock_blob.name = "test_prefix/test_file.txt"
    mock_storage_client.return_value.list_blobs.return_value = [mock_blob]

    with (
        patch("os.makedirs"),
        patch("builtins.print"),
        patch.object(mock_blob, "download_to_filename") as mock_download,
    ):
        download_gcs_dir_to_local(
            "gs://test-bucket/test_prefix", "/tmp/test_dir"
        )
        mock_download.assert_called_once_with("/tmp/test_dir/test_file.txt")


def test_download_gcs_dir_to_local_with_invalid_path() -> None:
    """Tests the download_gcs_dir_to_local function with an invalid path."""
    with pytest.raises(
        ValueError, match="is not a GCS path starting with gs://."
    ):
        download_gcs_dir_to_local("invalid_path", "/tmp/test_dir")
