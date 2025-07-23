import os
import tempfile
from unittest.mock import Mock, mock_open, patch
import pandas as pd
import numpy as np
import yaml

from trainer.vertex import create_vertex_ai_eval, write_model_schemas
from trainer.config import Config


class TestCreateVertexAIEval:
    """Test cases for the create_vertex_ai_eval function."""

    def setup_method(self):
        """Set up test data for each test method."""
        # Create sample binary classification data
        np.random.seed(42)
        n_samples = 100

        # Create true labels (binary: 0 and 1)
        self.y_true = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])

        # Create prediction scores that correlate with true labels
        self.y_scores = np.random.beta(
            a=np.where(
                self.y_true == 1, 3, 1
            ),  # Higher scores for positive class
            b=np.where(
                self.y_true == 1, 1, 3
            ),  # Lower scores for negative class
            size=n_samples,
        )

        # Create DataFrames
        self.df = pd.DataFrame(
            {
                "label": self.y_true,
                "feature1": np.random.randn(n_samples),
                "feature2": np.random.randn(n_samples),
            }
        )

        self.predictions = pd.DataFrame(
            {
                0: 1 - self.y_scores,  # Probability for class 0
                1: self.y_scores,  # Probability for class 1
            }
        )

        self.label_column = "label"
        self.positive_class = 1

    def test_create_vertex_ai_eval_basic_structure(self):
        """Test that the function returns the expected structure."""
        result = create_vertex_ai_eval(
            self.label_column, self.positive_class, self.df, self.predictions
        )

        # Check top-level keys
        assert "auPrc" in result
        assert "auRoc" in result
        assert "confidenceMetrics" in result
        assert "confusionMatrix" in result

        # Check data types
        assert isinstance(result["auPrc"], float)
        assert isinstance(result["auRoc"], float)
        assert isinstance(result["confidenceMetrics"], list)
        assert isinstance(result["confusionMatrix"], dict)

    def test_auc_metrics_valid_range(self):
        """Test that AUC metrics are in valid range [0, 1]."""
        result = create_vertex_ai_eval(
            self.label_column, self.positive_class, self.df, self.predictions
        )

        assert 0 <= result["auPrc"] <= 1
        assert 0 <= result["auRoc"] <= 1

    def test_confidence_metrics_structure(self):
        """Test the structure of confidence metrics."""
        result = create_vertex_ai_eval(
            self.label_column, self.positive_class, self.df, self.predictions
        )

        confidence_metrics = result["confidenceMetrics"]
        assert len(confidence_metrics) > 0

        # Test first metric structure
        metric = confidence_metrics[0]
        expected_keys = {
            "confidenceThreshold",
            "recall",
            "precision",
            "falsePositiveRate",
            "f1Score",
            "truePositiveCount",
            "falsePositiveCount",
            "trueNegativeCount",
            "falseNegativeCount",
        }
        assert expected_keys.issubset(metric.keys())

        # Check data types
        assert isinstance(metric["confidenceThreshold"], float)
        assert isinstance(metric["recall"], float)
        assert isinstance(metric["precision"], float)
        assert isinstance(metric["falsePositiveRate"], float)
        assert isinstance(metric["f1Score"], float)
        assert isinstance(metric["truePositiveCount"], int)
        assert isinstance(metric["falsePositiveCount"], int)
        assert isinstance(metric["trueNegativeCount"], int)
        assert isinstance(metric["falseNegativeCount"], int)

    def test_confidence_metrics_valid_ranges(self):
        """Test that confidence metrics are in valid ranges."""
        result = create_vertex_ai_eval(
            self.label_column, self.positive_class, self.df, self.predictions
        )

        for metric in result["confidenceMetrics"]:
            # Probabilities should be in [0, 1]
            assert 0 <= metric["recall"] <= 1
            assert 0 <= metric["precision"] <= 1
            assert 0 <= metric["falsePositiveRate"] <= 1
            assert 0 <= metric["f1Score"] <= 1

            # Counts should be non-negative
            assert metric["truePositiveCount"] >= 0
            assert metric["falsePositiveCount"] >= 0
            assert metric["trueNegativeCount"] >= 0
            assert metric["falseNegativeCount"] >= 0

    def test_confusion_matrix_structure(self):
        """Test the structure of the confusion matrix."""
        result = create_vertex_ai_eval(
            self.label_column, self.positive_class, self.df, self.predictions
        )

        cm = result["confusionMatrix"]
        assert "annotationSpecs" in cm
        assert "rows" in cm

        # Check annotation specs
        annotation_specs = cm["annotationSpecs"]
        assert len(annotation_specs) == 2
        assert "displayName" in annotation_specs[0]
        assert "displayName" in annotation_specs[1]

        # Check rows (2x2 matrix for binary classification)
        rows = cm["rows"]
        assert len(rows) == 2
        assert all(len(subrow) == 2 for row in rows for subrow in row.values())
        assert all(
            isinstance(cell, int)
            for row in rows
            for subrow in row.values()
            for cell in subrow
        )

    def test_confusion_matrix_in_confidence_metrics(self):
        """Test that confusion matrix appears in confidence metrics for 0.5 threshold."""
        result = create_vertex_ai_eval(
            self.label_column, self.positive_class, self.df, self.predictions
        )

        # Find the metric closest to 0.5 threshold
        metrics_with_cm = [
            m for m in result["confidenceMetrics"] if "confusionMatrix" in m
        ]

        # Check if 0.5 threshold exists in the thresholds
        thresholds = [
            m["confidenceThreshold"] for m in result["confidenceMetrics"]
        ]
        has_05_threshold = any(abs(t - 0.5) < 1e-6 for t in thresholds)

        if has_05_threshold:
            # Should have exactly one metric with confusion matrix (at 0.5)
            assert len(metrics_with_cm) == 1

            # Check structure of confusion matrix in confidence metrics
            cm_in_metric = metrics_with_cm[0]["confusionMatrix"]
            assert "annotationSpecs" in cm_in_metric
            assert "rows" in cm_in_metric
        else:
            # If no 0.5 threshold, that's also acceptable since thresholds come from the data
            # In this case, no confidence metrics should have confusion matrix
            assert len(metrics_with_cm) == 0

    def test_thresholds_sorted_descending(self):
        """Test that confidence thresholds are sorted in descending order."""
        result = create_vertex_ai_eval(
            self.label_column, self.positive_class, self.df, self.predictions
        )

        thresholds = [
            m["confidenceThreshold"] for m in result["confidenceMetrics"]
        ]
        assert thresholds == sorted(thresholds, reverse=True)

    def test_perfect_classifier(self):
        """Test with perfect classifier (should have AUC = 1.0)."""
        # Create perfect predictions
        perfect_df = pd.DataFrame({"label": [0, 0, 1, 1]})
        perfect_predictions = pd.DataFrame(
            {
                0: [
                    0.9,
                    0.8,
                    0.1,
                    0.2,
                ],  # High prob for class 0 when true label is 0
                1: [
                    0.1,
                    0.2,
                    0.9,
                    0.8,
                ],  # High prob for class 1 when true label is 1
            }
        )

        result = create_vertex_ai_eval(
            "label", 1, perfect_df, perfect_predictions
        )

        # Perfect classifier should have AUC close to 1.0
        assert result["auRoc"] > 0.95
        assert result["auPrc"] > 0.95

    def test_random_classifier(self):
        """Test with random classifier (should have AUC ≈ 0.5)."""
        # Create random predictions
        np.random.seed(123)
        random_df = pd.DataFrame({"label": np.random.choice([0, 1], 100)})
        random_predictions = pd.DataFrame(
            {0: np.random.random(100), 1: np.random.random(100)}
        )
        # Normalize to sum to 1
        total = random_predictions[0] + random_predictions[1]
        random_predictions[0] = random_predictions[0] / total
        random_predictions[1] = random_predictions[1] / total

        result = create_vertex_ai_eval(
            "label", 1, random_df, random_predictions
        )

        # Random classifier should have AUC around 0.5 (with some tolerance)
        assert 0.3 <= result["auRoc"] <= 0.7

    def test_edge_case_all_same_class(self):
        """Test with data where all samples belong to the same class."""
        # All positive class
        edge_df = pd.DataFrame({"label": [1, 1, 1, 1]})
        edge_predictions = pd.DataFrame(
            {0: [0.2, 0.3, 0.1, 0.4], 1: [0.8, 0.7, 0.9, 0.6]}
        )

        # This should not crash and should handle the edge case gracefully
        result = create_vertex_ai_eval("label", 1, edge_df, edge_predictions)

        assert "auPrc" in result
        assert "auRoc" in result
        assert "confidenceMetrics" in result
        assert "confusionMatrix" in result

    def test_string_labels(self):
        """Test with string labels instead of numeric."""
        string_df = pd.DataFrame(
            {"label": ["negative", "positive", "negative", "positive"]}
        )
        string_predictions = pd.DataFrame(
            {"negative": [0.8, 0.2, 0.7, 0.3], "positive": [0.2, 0.8, 0.3, 0.7]}
        )

        result = create_vertex_ai_eval(
            "label", "positive", string_df, string_predictions
        )

        assert isinstance(result["auPrc"], float)
        assert isinstance(result["auRoc"], float)
        assert len(result["confidenceMetrics"]) > 0

    def test_f1_score_calculation(self):
        """Test that F1 score is calculated correctly."""
        result = create_vertex_ai_eval(
            self.label_column, self.positive_class, self.df, self.predictions
        )

        for metric in result["confidenceMetrics"]:
            precision = metric["precision"]
            recall = metric["recall"]
            f1_score = metric["f1Score"]

            if precision + recall > 0:
                expected_f1 = 2 * precision * recall / (precision + recall)
                assert abs(f1_score - expected_f1) < 1e-10
            else:
                assert f1_score == 0.0

    def test_confusion_matrix_counts_consistency(self):
        """Test that confusion matrix counts are consistent with total sample size."""
        result = create_vertex_ai_eval(
            self.label_column, self.positive_class, self.df, self.predictions
        )

        for metric in result["confidenceMetrics"]:
            total_predicted = (
                metric["truePositiveCount"]
                + metric["falsePositiveCount"]
                + metric["trueNegativeCount"]
                + metric["falseNegativeCount"]
            )
            assert total_predicted == len(self.df)

    def test_confusion_matrix_at_specific_threshold(self):
        """Test that confusion matrix is added for 0.5 threshold when it exists."""
        # Create data where we're likely to get a 0.5 threshold
        simple_df = pd.DataFrame({"label": [0, 0, 1, 1]})
        simple_predictions = pd.DataFrame(
            {
                0: [0.8, 0.6, 0.4, 0.2],  # Scores for class 0
                1: [
                    0.2,
                    0.4,
                    0.6,
                    0.8,
                ],  # Scores for class 1 (includes 0.5-ish values)
            }
        )

        result = create_vertex_ai_eval(
            "label", 1, simple_df, simple_predictions
        )

        # Check that we have confidence metrics
        assert len(result["confidenceMetrics"]) > 0

        # The confusion matrix should always be present at the top level
        assert "confusionMatrix" in result
        assert "annotationSpecs" in result["confusionMatrix"]
        assert "rows" in result["confusionMatrix"]


class TestWriteModelSchemas:
    """Test cases for the write_model_schemas function."""

    def setup_method(self):
        """Set up test data for each test method."""
        # Create mock config
        self.config = Mock(spec=Config)
        self.config.model_export_uri = "/tmp/model_export"

        # Create mock predictor with feature metadata and class labels
        self.predictor = Mock()

        # Mock feature metadata - simulating different data types
        mock_feature_metadata = Mock()
        mock_feature_metadata.to_dict.return_value = {
            "feature_int": ("int", ()),
            "feature_float": ("float", ()),
            "feature_string": ("object", ()),
            "feature_category": ("category", ()),
            "feature_bool": ("bool", ()),
            "feature_datetime": ("datetime", ()),
        }
        self.predictor.feature_metadata_in = mock_feature_metadata

        # Mock class labels for binary classification
        self.predictor.class_labels = ["class_0", "class_1"]

    @patch("trainer.vertex.gcs_path")
    @patch("trainer.vertex.os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    @patch("trainer.vertex.yaml.dump")
    def test_write_model_schemas_success(
        self, mock_yaml_dump, mock_file_open, mock_makedirs, mock_gcs_path
    ):
        """Test successful schema writing with mocked file operations."""
        # Setup mocks
        mock_gcs_path.side_effect = lambda base, filename: f"{base}/{filename}"

        # Call the function
        write_model_schemas(self.config, self.predictor)

        # Verify gcs_path was called for each schema file
        expected_calls = [
            ("/tmp/model_export", "instance_schema.yaml"),
            ("/tmp/model_export", "prediction_schema.yaml"),
            ("/tmp/model_export", "parameters_schema.yaml"),
        ]
        assert mock_gcs_path.call_count == 3
        actual_calls = [c.args for c in mock_gcs_path.call_args_list]
        for call in expected_calls:
            assert call in actual_calls

        # Verify makedirs was called for each schema directory
        assert mock_makedirs.call_count == 3
        mock_makedirs.assert_any_call("/tmp/model_export", exist_ok=True)

        # Verify files were opened for writing
        assert mock_file_open.call_count == 3
        expected_file_paths = [
            "/tmp/model_export/instance_schema.yaml",
            "/tmp/model_export/prediction_schema.yaml",
            "/tmp/model_export/parameters_schema.yaml",
        ]
        for path in expected_file_paths:
            mock_file_open.assert_any_call(path, "w")

        # Verify yaml.dump was called 3 times with correct parameters
        assert mock_yaml_dump.call_count == 3

        # Get the schemas that were written
        yaml_dump_calls = mock_yaml_dump.call_args_list
        schemas_written = [call[0][0] for call in yaml_dump_calls]

        # Verify instance schema structure
        instance_schema = next(
            s for s in schemas_written if s.get("title") == "InstanceSchema"
        )
        assert instance_schema["type"] == "object"
        assert "properties" in instance_schema
        assert "required" in instance_schema
        assert "example" in instance_schema

        # Check that all features are in the schema
        expected_features = [
            "feature_int",
            "feature_float",
            "feature_string",
            "feature_category",
            "feature_bool",
            "feature_datetime",
        ]
        for feature in expected_features:
            assert feature in instance_schema["properties"]
            assert feature in instance_schema["required"]
            assert feature in instance_schema["example"]

        # Verify feature type mappings
        assert instance_schema["properties"]["feature_int"]["type"] == "integer"
        assert instance_schema["properties"]["feature_int"]["format"] == "int64"
        assert (
            instance_schema["properties"]["feature_float"]["type"] == "number"
        )
        assert (
            instance_schema["properties"]["feature_float"]["format"] == "float"
        )
        assert (
            instance_schema["properties"]["feature_string"]["type"] == "string"
        )
        assert "format" not in instance_schema["properties"]["feature_string"]
        assert (
            instance_schema["properties"]["feature_bool"]["type"] == "boolean"
        )
        assert (
            instance_schema["properties"]["feature_datetime"]["type"]
            == "string"
        )
        assert (
            instance_schema["properties"]["feature_datetime"]["format"]
            == "date-time"
        )

        # Verify prediction schema structure
        prediction_schema = next(
            s for s in schemas_written if s.get("title") == "PredictionResponse"
        )
        assert prediction_schema["title"] == "PredictionResponse"
        assert "oneOf" in prediction_schema
        assert len(prediction_schema["oneOf"]) == 2

        # Check array format (first option)
        array_format = prediction_schema["oneOf"][0]
        assert array_format["type"] == "array"
        assert array_format["items"]["type"] == "number"
        assert array_format["items"]["format"] == "float"
        assert array_format["items"]["minimum"] == 0.0
        assert array_format["items"]["maximum"] == 1.0

        # Check object format (second option)
        object_format = prediction_schema["oneOf"][1]
        assert object_format["type"] == "object"
        assert "class_0" in object_format["properties"]
        assert "class_1" in object_format["properties"]
        assert "class_0" in object_format["required"]
        assert "class_1" in object_format["required"]

        # Verify parameters schema structure
        parameters_schema = next(
            s
            for s in schemas_written
            if s.get("type") == "object"
            and "as_object" in s.get("properties", {})
        )
        assert parameters_schema["type"] == "object"
        assert "as_object" in parameters_schema["properties"]
        assert parameters_schema["properties"]["as_object"]["type"] == "boolean"
        assert "description" in parameters_schema["properties"]["as_object"]

        # Verify yaml.dump was called with sort_keys=False
        for call in yaml_dump_calls:
            assert call[1]["sort_keys"] is False

    @patch("trainer.vertex.gcs_path")
    @patch("trainer.vertex.os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    @patch("trainer.vertex.yaml.dump")
    @patch("trainer.vertex.logging.error")
    def test_write_model_schemas_exception_handling(
        self,
        mock_logging_error,
        mock_yaml_dump,
        mock_file_open,
        mock_makedirs,
        mock_gcs_path,
    ):
        """Test that exceptions are properly handled and logged."""
        # Setup mocks to raise an exception
        mock_gcs_path.side_effect = Exception("Test exception")

        # Call the function
        result = write_model_schemas(self.config, self.predictor)

        # Verify the function returns None on exception
        assert result is None

        # Verify error was logged
        mock_logging_error.assert_called_once()
        error_call = mock_logging_error.call_args
        assert "Error writing schemas:" in error_call[0][0]
        assert error_call[1]["exc_info"] is not None

    @patch("trainer.vertex.gcs_path")
    @patch("trainer.vertex.os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    @patch("trainer.vertex.yaml.dump")
    def test_write_model_schemas_with_multiclass(
        self, mock_yaml_dump, mock_file_open, mock_makedirs, mock_gcs_path
    ):
        """Test schema writing with multiclass classification."""
        # Setup mocks
        mock_gcs_path.side_effect = lambda base, filename: f"{base}/{filename}"

        # Set up multiclass labels
        self.predictor.class_labels = [
            "class_a",
            "class_b",
            "class_c",
            "class_d",
        ]

        # Call the function
        write_model_schemas(self.config, self.predictor)

        # Get the prediction schema that was written
        yaml_dump_calls = mock_yaml_dump.call_args_list
        schemas_written = [call[0][0] for call in yaml_dump_calls]
        prediction_schema = next(
            s for s in schemas_written if s.get("title") == "PredictionResponse"
        )

        # Verify all classes are in the object format schema
        object_format = prediction_schema["oneOf"][1]
        for class_label in ["class_a", "class_b", "class_c", "class_d"]:
            assert class_label in object_format["properties"]
            assert class_label in object_format["required"]
            assert object_format["properties"][class_label]["type"] == "number"
            assert object_format["properties"][class_label]["format"] == "float"
            assert object_format["properties"][class_label]["minimum"] == 0.0
            assert object_format["properties"][class_label]["maximum"] == 1.0

    @patch("trainer.vertex.gcs_path")
    @patch("trainer.vertex.os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    @patch("trainer.vertex.yaml.dump")
    def test_write_model_schemas_with_numeric_labels(
        self, mock_yaml_dump, mock_file_open, mock_makedirs, mock_gcs_path
    ):
        """Test schema writing with numeric class labels."""
        # Setup mocks
        mock_gcs_path.side_effect = lambda base, filename: f"{base}/{filename}"

        # Set up numeric labels
        self.predictor.class_labels = [0, 1, 2]

        # Call the function
        write_model_schemas(self.config, self.predictor)

        # Get the prediction schema that was written
        yaml_dump_calls = mock_yaml_dump.call_args_list
        schemas_written = [call[0][0] for call in yaml_dump_calls]
        prediction_schema = next(
            s for s in schemas_written if s.get("title") == "PredictionResponse"
        )

        # Verify numeric labels are converted to strings in the schema
        object_format = prediction_schema["oneOf"][1]
        for class_label in ["0", "1", "2"]:
            assert class_label in object_format["properties"]
            assert class_label in object_format["required"]

    @patch("trainer.vertex.gcs_path")
    @patch("trainer.vertex.os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    @patch("trainer.vertex.yaml.dump")
    def test_write_model_schemas_edge_case_features(
        self, mock_yaml_dump, mock_file_open, mock_makedirs, mock_gcs_path
    ):
        """Test schema writing with edge case feature types."""
        # Setup mocks
        mock_gcs_path.side_effect = lambda base, filename: f"{base}/{filename}"

        # Mock feature metadata with unknown type
        mock_feature_metadata = Mock()
        mock_feature_metadata.to_dict.return_value = {
            "unknown_type_feature": ("unknown_type", ()),
            "text_feature": ("text", ()),
        }
        self.predictor.feature_metadata_in = mock_feature_metadata

        # Call the function
        write_model_schemas(self.config, self.predictor)

        # Get the instance schema that was written
        yaml_dump_calls = mock_yaml_dump.call_args_list
        schemas_written = [call[0][0] for call in yaml_dump_calls]
        instance_schema = next(
            s for s in schemas_written if s.get("title") == "InstanceSchema"
        )

        # Verify unknown types default to string
        assert (
            instance_schema["properties"]["unknown_type_feature"]["type"]
            == "string"
        )
        assert (
            "format"
            not in instance_schema["properties"]["unknown_type_feature"]
        )

        # Verify text type is handled correctly
        assert instance_schema["properties"]["text_feature"]["type"] == "string"
        assert "format" not in instance_schema["properties"]["text_feature"]
