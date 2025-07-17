import pandas as pd
import pytest
from unittest.mock import MagicMock
from predictor.utils import parse_instances_to_dataframe


# Mock TabularPredictor for testing
class MockFeatureMetadata:
    def to_dict(self):
        return {"feature1": "int", "feature2": "float", "feature3": "object"}


class MockTabularPredictor:
    def __init__(self):
        self.feature_metadata_in = MockFeatureMetadata()


@pytest.fixture
def mock_predictor():
    return MockTabularPredictor()


def test_parse_instances_with_list_of_dicts(mock_predictor):
    instances = [
        {"feature1": 1, "feature2": 2.0, "feature3": "a"},
        {"feature1": 2, "feature2": 3.0, "feature3": "b"},
    ]
    df = parse_instances_to_dataframe(instances, mock_predictor)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 3)
    assert list(df.columns) == ["feature1", "feature2", "feature3"]
    assert df["feature1"].tolist() == [1, 2]


def test_parse_instances_with_list_of_lists(mock_predictor):
    instances = [
        [1, 2.0, "a"],
        [2, 3.0, "b"],
    ]
    df = parse_instances_to_dataframe(instances, mock_predictor)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 3)
    assert list(df.columns) == ["feature1", "feature2", "feature3"]
    assert df["feature1"].tolist() == [1, 2]


def test_parse_instances_with_missing_features_in_dict(mock_predictor):
    instances = [{"feature1": 1}]
    with pytest.raises(ValueError, match="Missing required features"):
        parse_instances_to_dataframe(instances, mock_predictor)


def test_parse_instances_with_wrong_number_of_features_in_list(mock_predictor):
    instances = [[1, 2.0]]
    with pytest.raises(
        ValueError, match="Instance 0 has 2 values, but 3 are expected."
    ):
        parse_instances_to_dataframe(instances, mock_predictor)


def test_parse_instances_with_empty_list(mock_predictor):
    instances = []
    df = parse_instances_to_dataframe(instances, mock_predictor)
    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert list(df.columns) == ["feature1", "feature2", "feature3"]


def test_parse_instances_with_invalid_format(mock_predictor):
    instances = ["invalid_instance"]
    with pytest.raises(ValueError, match="Invalid instances format"):
        parse_instances_to_dataframe(instances, mock_predictor)  # type: ignore


def test_parse_instances_extra_features_in_dict(mock_predictor):
    instances = [
        {"feature1": 1, "feature2": 2.0, "feature3": "a", "extra": "value"},
    ]
    df = parse_instances_to_dataframe(instances, mock_predictor)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, 3)
    assert "extra" not in df.columns
    assert list(df.columns) == ["feature1", "feature2", "feature3"]
