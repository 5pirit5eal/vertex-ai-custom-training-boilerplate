"""Tests for the trainer.experiment module."""

from unittest.mock import MagicMock, patch

from trainer.experiment import log_nested_metrics, log_learning_curves


@patch("trainer.experiment.aiplatform")
def test_log_nested_metrics(mock_aiplatform):
    """Tests that nested metrics are logged correctly."""
    metrics = {"level1": {"level2": {"metric1": 0.9, "metric2": 0.8}}}
    log_nested_metrics(metrics)
    mock_aiplatform.log_metrics.assert_any_call({"level1 level2 metric1": 0.9})
    mock_aiplatform.log_metrics.assert_any_call({"level1 level2 metric2": 0.8})


@patch("trainer.experiment.aiplatform")
@patch("trainer.experiment.time.sleep", return_value=None)
def test_log_learning_curves(mock_sleep, mock_aiplatform):
    """Tests that learning curves are logged correctly."""
    model_data = {
        "model1": [
            ["train", "val"],
            ["accuracy"],
            [[[0.1, 0.2], [0.3, 0.4]]],
        ]
    }
    log_learning_curves(model_data)
    mock_aiplatform.log_time_series_metrics.assert_any_call(
        {"model1 accuracy train": 0.1, "model1 accuracy val": 0.3}, step=0
    )
    mock_aiplatform.log_time_series_metrics.assert_any_call(
        {"model1 accuracy train": 0.2, "model1 accuracy val": 0.4}, step=1
    )
