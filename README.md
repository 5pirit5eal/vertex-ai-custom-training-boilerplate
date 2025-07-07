# AutoGluon Vertex AI Custom Training & Prediction Boilerplate

This repository provides a production-ready boilerplate for training and serving machine learning models using AutoGluon on Google Cloud's Vertex AI. AutoGluon enables **automatic machine learning** in just a few lines of code, making it ideal for building high-performance tabular models quickly.

## 🎯 Key Features

- **Vertex AI Dataset Integration**: Native support for Vertex AI managed datasets as custom job inputs
- **Vertex AI Experiments Integration**: Configurable use of Vertex AI Experiments
- **Flexible Configuration**: Both environment variables and command-line arguments supported (args preferred)
- **GPU Acceleration**: Full CUDA support for deep learning models
- **Multiple Data Sources**: BigQuery, CSV files, and Cloud Storage with wildcard patterns

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Quick Start Guide](#quick-start-guide)
- [Configuration](#configuration)
- [Data Formats](#data-formats)
- [Training Job Setup](#training-job-setup)
- [Prediction Service](#prediction-service)
- [Deployment](#deployment)
- [Monitoring & Logging](#monitoring--logging)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

## 🎯 Overview

- **Custom Training**: Containerized training jobs that can handle large datasets from BigQuery or Cloud Storage
- **Model Serving**: REST API prediction service with health checks and auto-scaling
- **Production Ready**: GPU/CPU support, experiment tracking, logging, and deployment optimization
- **Flexible Configuration**: Support for multiple data formats, hyperparameter tuning, and experiment tracking

### What is AutoGluon?

[AutoGluon](https://auto.gluon.ai/) is an open-source AutoML framework that delivers **state-of-the-art predictive performance** with minimal effort. It:

- **Automatically** selects optimal models, features, and hyperparameters
- Supports **tabular**, **multimodal** (text + tabular), and **time series** data
- Provides **ensemble learning** with multiple algorithms (XGBoost, Neural Networks, LightGBM, etc.)
- Offers **production-ready models** with fast inference and deployment optimization

## ✨ Features

### Training Component

- 🔄 **Automated ML Pipeline**: Ensemble learning with 10+ algorithms
- 📊 **Multi-format Data Support**: CSV, BigQuery, with wildcard file patterns
- ⚡ **GPU Acceleration**: CUDA support for neural networks and deep learning models
- 🎯 **Hyperparameter Optimization**: Automated search with time-based limits
- 📈 **Experiment Tracking**: Integration with Vertex AI Experiments and TensorBoard
- 🔧 **Flexible Configuration**: 30+ configurable parameters via args and environment variables
- 💾 **Model Checkpointing**: Resume training from checkpoints
- 📏 **Feature Engineering**: Automatic feature selection and importance calculation

### Prediction Component

- 🌐 **REST API**: Litestar-based high-performance web server
- 🏥 **Health Checks**: Built-in readiness and liveness probes
- 🔧 **Auto-scaling**: Compatible with Vertex AI Endpoints auto-scaling
- 💨 **Fast Inference**: Model persistence and batch prediction support
- 🔒 **Production Security**: CUDA support and optimized Docker images
- ☁️ **Cloud Storage**: Automatic model download from GCS buckets

## 🏗️ Architecture

```text
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources  │    │  Training Job    │    │ Model Registry  │
│                 │    │                  │    │                 │
│ • BigQuery      │───▶│ • AutoGluon      │───▶│ • GCS Bucket    │
│ • CSV Files     │    │ • GPU/CPU        │    │ • Model Files   │
│ • GCS Buckets   │    │ • Experiments    │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                      ┌───────────────────┐     ┌──────────────────┐
                      │ Vertex Experiment │     │ Prediction API   │
                      │                   │     │                  │
                      │ • Metrics         │     │ • REST Endpoint  │
                      │ • Parameters      │     │ • Health Checks  │
                      │ • Artifacts       │     │ • Auto-scaling   │
                      └───────────────────┘     └──────────────────┘
```

## 📁 Project Structure

```text
vertex-ai-custom-training-boilerplate/
├── build_and_push.sh              # Docker build & push script
├── README.md                       # This documentation
├── trainer/                        # Training component
│   ├── Dockerfile                  # Multi-stage Docker build
│   ├── pyproject.toml             # Dependencies & configuration
│   ├── uv.lock                    # Locked dependencies
│   └── src/trainer/
│       ├── main.py                # Training entry point
│       ├── config.py              # Configuration management
│       └── data.py                # Data loading, logging & processing
└── predictor/                     # Prediction component
    ├── Dockerfile                 # Multi-stage Docker build
    ├── pyproject.toml            # Dependencies & configuration
    ├── uv.lock                   # Locked dependencies
    └── src/predictor/
        ├── main.py               # Prediction server
        └── utils.py              # GCS utilities
```

### Key Files

| File | Purpose | Important Features |
|------|---------|-------------------|
| `trainer/src/trainer/main.py` | Training orchestration | AutoGluon fitting, experiment tracking, model export |
| `trainer/src/trainer/config.py` | Configuration management | 30+ parameters, validation, CLI interface |
| `trainer/src/trainer/data.py` | Data processing | BigQuery/CSV loading, feature preprocessing |
| `predictor/src/predictor/main.py` | Prediction API | Litestar server, health checks, batch inference |
| `build_and_push.sh` | Deployment automation | Docker build, Artifact Registry push |

## 🔧 Prerequisites

### Google Cloud Setup

1. **Google Cloud Project** with billing enabled
2. **APIs Enabled**:

   ```bash
   gcloud services enable aiplatform.googleapis.com
   gcloud services enable artifactregistry.googleapis.com
   gcloud services enable storage.googleapis.com
   gcloud services enable bigquery.googleapis.com
   ```

3. **Artifact Registry Repository**:

   ```bash
   gcloud artifacts repositories create my-repo \
     --repository-format=docker \
     --location=europe-west3
   ```

4. **IAM Permissions** for the Vertex AI service account:
   - `roles/storage.objectAdmin`
   - `roles/bigquery.dataViewer` (if using BigQuery)
   - `roles/aiplatform.user`

### Local Development

- **Docker** (for building images)
- **Google Cloud SDK** (gcloud CLI)
- **Python 3.12+** (for local development)

## 🚀 Quick Start Guide

### 1. Clone and Configure

```bash
git clone <repository-url>
cd vertex-ai-custom-training-boilerplate

# Make build script executable
chmod +x build_and_push.sh
```

### 2. Build and Push Images

```bash
# Build trainer image
./build_and_push.sh trainer YOUR_PROJECT_ID europe-west3 my-repo

# Build predictor image  
./build_and_push.sh predictor YOUR_PROJECT_ID europe-west3 my-repo
```

### 3. Prepare Your Data

#### Option A: Vertex AI Datasets (Recommended) ✨

```python
# Create a Vertex AI dataset (most seamless approach)
dataset = aiplatform.TabularDataset.create(
    display_name="my-training-data",
    bq_source="bq://your-project.dataset.table",  # or gcs_source for CSV
    labels={"environment": "production"}
)
```

#### Option B: CSV Files in Cloud Storage

```bash
# Upload your training data
gsutil cp train.csv gs://your-bucket/data/
gsutil cp test.csv gs://your-bucket/data/
```

#### Option C: BigQuery Tables

```sql
-- Your data should be in BigQuery tables
-- Example: project.dataset.train_table
```

### 4. Run Training Job (Modern Approach)

```python
from google.cloud import aiplatform

OUTPUT_DIR="gs://your-bucket/experiment-1"

# Initialize Vertex AI
aiplatform.init(
    project="your-project-id",
    location="europe-west3"
)

# Create custom training job
job = aiplatform.CustomContainerTrainingJob(
    display_name="autogluon-training",
    container_uri="europe-west3-docker.pkg.dev/your-project/my-repo/autogluon-train:latest",
    model_serving_container_image_uri="europe-west3-docker.pkg.dev/your-project/my-repo/autogluon-serve:latest",
    model_serving_container_predict_route="/predict",
    model_serving_container_health_route="/ping",
    model_serving_container_ports=[8080],
    model_instance_schema_uri=OUTPUT_DIR + "/model/instance_schema.yaml",
    model_prediction_schema_uri=OUTPUT_DIR + "/model/prediction_schema.yaml"
)

# Option A: Using Vertex AI Dataset (Recommended)
model = job.run(
    dataset=dataset,  # Uses the dataset created above
    base_output_dir=OUTPUT_DIR,
    replica_count=1,
    machine_type="n1-highmem-16",
    args=[
        "--label", "target",
        "--presets", "best_quality",
        "--time-limit", "3600", # in seconds
        "--experiment-name", "my-experiment"
    ]   
)

# Option B: Using direct data URIs
model = job.run(
    base_output_dir=OUTPUT_DIR,
    replica_count=1,
    machine_type="n1-highmem-16", 
    args=[
        "--data-format", "csv",
        "--train-data-uri", "gs://your-bucket/data/train.csv",
        "--test-data-uri", "gs://your-bucket/data/test.csv",
        "--label", "target",
        "--presets", "best_quality"
    ]
)
```

### 5. Deploy for Predictions

```python
# Deploy the trained model
endpoint = model.deploy(
    machine_type="n1-highmem-8",
    min_replica_count=1,
    max_replica_count=3
)

# Make predictions
predictions = endpoint.predict(instances=[
    {"feature1": 1.0, "feature2": "value", "feature3": 42}
])
```

## ⚙️ Configuration

The training component supports configuration through both **environment variables** and **command-line arguments**. **Command-line arguments are preferred** as they provide better validation and easier debugging.

> **💡 Tip**: Use command-line arguments when possible. Environment variables are mainly for compatibility with Vertex AI's predefined variables.

### Data Input Options

#### Vertex AI Datasets ✨
**NEW**: Native support for Vertex AI managed datasets as custom job inputs:

```python
# Using Vertex AI Dataset resource
dataset = aiplatform.TabularDataset("projects/PROJECT/locations/REGION/datasets/DATASET_ID")
job.run(
    dataset=dataset,  # Automatically sets data URIs
    args=[
        "--label", "target_column",
        "--presets", "best_quality"
    ]
)
```

#### Manual Data Sources
For custom data sources (BigQuery, CSV files):

```python
job.run(
    args=[
        "--data-format", "csv",
        "--train-data-uri", "gs://bucket/train.csv",
        "--label", "target_column"
    ]
)
```

### Core Parameters

| Parameter | Type | CLI Argument | Environment Variable | Description |
|-----------|------|--------------|---------------------|-------------|
| Data format | Required | `--data-format` | `AIP_DATA_FORMAT` | `csv` or `bigquery` |
| Training data | Required | `--train-data-uri` | `AIP_TRAINING_DATA_URI` | Path to training data or Vertex AI dataset |
| Validation data | Optional | `--val-data-uri` | `AIP_VALIDATION_DATA_URI` | Validation dataset path |
| Test data | Optional | `--test-data-uri` | `AIP_TEST_DATA_URI` | Test dataset path |
| Label column | Required | `--label` | `AIP_LABEL_COLUMN` | Target column name for prediction |
| Weight column | Optional | `--weight-column` | - | Sample weight column name |
| Training time | Optional | `--time-limit` | - | Training time limit in seconds (default: auto) |
| Quality preset | Optional | `--presets` | - | `best_quality`, `high_quality`, `good_quality`, `medium_quality` |
| Task type | Optional | `--task-type` | - | `binary`, `multiclass`, `regression`, `quantile` (auto-detected) |
| Evaluation metric | Optional | `--eval-metric` | - | `accuracy`, `roc_auc`, `rmse`, etc. |
| GPU usage | Flag | `--use-gpu` | - | Enable GPU acceleration |
| Multimodal | Flag | `--multimodal` | - | Enable text + tabular features |
| Feature importance | Flag | `--calculate-importance` | - | Calculate feature importance |
| Model directory | Required | `--model-export-uri` | `AIP_MODEL_DIR` | Output directory for trained model |
| Checkpoint directory | Optional | `--checkpoint-uri` | `AIP_CHECKPOINT_DIR` | Checkpoint storage path |
| TensorBoard logs | Optional | `--tensorboard-log-uri` | `AIP_TENSORBOARD_LOG_DIR` | TensorBoard logging directory |
| Experiment name | Optional | `--experiment-name` | - | Vertex AI experiment name |
| Run name | Optional | `--experiment-run-name` | - | Specific run identifier |
| Model import | Optional | `--model-import-uri` | - | Resume from existing model |
| Project ID | Optional | `--project-id` | `CLOUD_ML_PROJECT_ID` | Google Cloud project ID |
| Region | Optional | `--region` | `CLOUD_ML_REGION` | Google Cloud region |
| Log level | Optional | `--log-level` | `CLOUD_ML_LOG_LEVEL` | Logging verbosity |

### Configuration Examples

#### Using Command-Line Arguments (Recommended)

```python
job.run(
    args=[
        "--data-format", "csv",
        "--train-data-uri", "gs://my-bucket/train.csv",
        "--val-data-uri", "gs://my-bucket/val.csv", 
        "--test-data-uri", "gs://my-bucket/test.csv",
        "--label", "target",
        "--presets", "best_quality",
        "--time-limit", "7200",
        "--use-gpu",
        "--calculate-importance",
        "--experiment-name", "customer-churn-v1"
    ]
)
```

## 📊 Data Formats & Sources

This boilerplate supports multiple data input methods to accommodate various ML workflows:

### 1. Vertex AI Managed Datasets ✨ **RECOMMENDED**

**Native support for Vertex AI datasets as custom job inputs** - the most seamless integration with Google Cloud ML workflow:

```python
# Create or use existing Vertex AI dataset
dataset = aiplatform.TabularDataset("projects/PROJECT/locations/REGION/datasets/DATASET_ID")

# Or create from data source
dataset = aiplatform.TabularDataset.create(
    display_name="customer-data",
    bq_source="bq://project.dataset.table",
    labels={"environment": "production"}
)

# Use dataset directly in training job
job.run(
    dataset=dataset,  # Vertex AI automatically handles data URIs
    args=[
        "--label", "target_column",
        "--presets", "best_quality",
        "--time-limit", "3600"
    ]
)
```

**Benefits of Vertex AI Datasets:**

- **Built-in data validation** and schema detection
- **Seamless integration** with Vertex AI Pipelines
- **Version control** for datasets
- **Automatic train/validation/test splits** (if configured)

### 2. CSV Files in Cloud Storage

#### Single File

```bash
# Command-line argument (preferred)
--train-data-uri gs://your-bucket/train.csv

# Environment variable (legacy)
export AIP_TRAINING_DATA_URI="gs://your-bucket/train.csv"
```

#### Multiple Files with Wildcards

```bash
# Load all CSV files matching pattern
--train-data-uri gs://your-bucket/training-*.csv
--train-data-uri gs://your-bucket/data/2024/*/train.csv
```

**File Format Requirements:**

- CSV with header row
- Target column specified in `--label` parameter
- Missing values: use empty cells or standard indicators (`NA`, `NULL`, etc.)
- Text features: raw text supported with `--multimodal` flag
- Sample weights: optional weight column via `--weight-column`

**Example CSV Structure:**
```csv
customer_id,age,income,category,text_review,target
1001,25,50000,premium,"Great service",1
1002,34,75000,standard,"Average experience",0
1003,45,95000,premium,"Excellent support",1
```

### 3. BigQuery Tables

```bash
# Direct table reference
--data-format bigquery
--train-data-uri bq://project.dataset.train_table
--val-data-uri bq://project.dataset.val_table
--test-data-uri bq://project.dataset.test_table
```

**BigQuery Features:**

- **Large-scale data processing** - handle terabytes of data
- **Real-time data integration** with streaming inserts
- **SQL-based feature engineering** in views
- **Automatic data partitioning** for performance

**Example BigQuery Setup:**
```sql
-- Create training view with feature engineering
CREATE VIEW `project.dataset.train_view` AS
SELECT 
  customer_id,
  age,
  income,
  category,
  EXTRACT(DAYOFWEEK FROM created_date) as day_of_week,
  target
FROM `project.dataset.raw_data`
WHERE split = 'train'
```

### GPU-Accelerated Training

```python
job = aiplatform.CustomContainerTrainingJob(
    display_name="autogluon-gpu-training",
    container_uri="europe-west3-docker.pkg.dev/PROJECT/REPO/autogluon-train:latest"
)

model = job.run(
    replica_count=1,
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
    args=[
        "--data-format", "csv",
        "--train-data-uri", "gs://bucket/large_dataset.csv",
        "--model-export-uri", "gs://bucket/models/",
        "--label", "target",
        "--use-gpu",
        "--time-limit", "7200",
        "--presets", "best_quality"
    ]
)
```

### Experiment Tracking & Advanced Features

```python
model = job.run(
    args=[
        "--train-data-uri", "gs://bucket/train.csv",
        "--val-data-uri", "gs://bucket/val.csv",
        "--model-export-uri", "gs://bucket/models/",
        "--label", "target",
        "--experiment-name", "customer-churn-prediction",
        "--experiment-run-name", "experiment-001",
        "--tensorboard-log-uri", "gs://bucket/experiments/",
        "--calculate-importance",
        "--presets", "best_quality",
        "--task-type", "binary",
        "--eval-metric", "f1"
    ]
)
```

## 🔮 Prediction Service

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ping` | GET | Health check (returns "Model is ready") |
| `/predict` | POST | Make predictions on batch data |

### Prediction Request Format

```json
{
  "instances": [
    {
      "feature1": 1.0,
      "feature2": "category_a", 
      "feature3": 42
    },
    {
      "feature1": 2.0,
      "feature2": "category_b",
      "feature3": 37
    }
  ]
}
```

### Prediction Response Format

```json
{
  "predictions": [
    {
      "class_0": 0.23,
      "class_1": 0.77
    },
    {
      "class_0": 0.84,
      "class_1": 0.16  
    }
  ]
}
```

### Local Testing

```python
import requests

# Health check
response = requests.get("http://localhost:8501/ping")
print(response.text)  # "Model is ready"

# Predictions
data = {
    "instances": [
        {"age": 25, "income": 50000, "category": "A"}
    ]
}
response = requests.post("http://localhost:8501/predict", json=data)
print(response.json())
```

## 🚀 Deployment

### Build Images

The `build_and_push.sh` script automates the Docker build and push process:

```bash
# Build trainer
./build_and_push.sh trainer PROJECT_ID REGION ARTIFACT_REGISTRY

# Build predictor  
./build_and_push.sh predictor PROJECT_ID REGION ARTIFACT_REGISTRY
```

**Script Features**:

- Multi-stage Docker builds for optimized images
- Automatic tagging with date and "latest"
- Platform-specific builds (linux/amd64)
- Artifact Registry integration

### Manual Docker Commands

```bash
# Build trainer
cd trainer
docker build --platform linux/amd64 -t autogluon-train .
docker tag autogluon-train europe-west3-docker.pkg.dev/PROJECT/REPO/autogluon-train:latest
docker push europe-west3-docker.pkg.dev/PROJECT/REPO/autogluon-train:latest

# Build predictor
cd predictor  
docker build --platform linux/amd64 -t autogluon-serve .
docker tag autogluon-serve europe-west3-docker.pkg.dev/PROJECT/REPO/autogluon-serve:latest
docker push europe-west3-docker.pkg.dev/PROJECT/REPO/autogluon-serve:latest
```

### Vertex AI Endpoint Deployment

```python
# Deploy with auto-scaling
endpoint = model.deploy(
    deployed_model_display_name="autogluon-predictor",
    machine_type="n1-highmem-8",
    min_replica_count=1,
    max_replica_count=10,
    traffic_percentage=100
)
```

## 📊 Monitoring & Logging

### Training Logs

Training logs are automatically saved to:

- **Console Output**: Visible in Vertex AI Training UI
- **TensorBoard**: Learning curves
- **Vertex AI Experiments**: Learning curves, metrics, parameters, artifacts
- **Log Files**: Saved to `$AIP_TENSORBOARD_LOG_DIR/training.log`

### Prediction Metrics

The prediction service provides:
- **Health Checks**: `/ping` endpoint
- **Request Logging**: All predictions logged
- **Error Handling**: Graceful error responses
- **Performance Metrics**: Request latency tracking

### Vertex AI Integration

- **Experiments**: Track training runs and hyperparameters
- **Model Registry**: Automatic model versioning
- **Endpoint Monitoring**: Request/response logging
- **TensorBoard**: Visualization of training metrics

## 🔧 Troubleshooting

### Common Issues

#### Issue: "Model not ready" error

```bash
# Check if model loading completed
curl http://your-endpoint/ping
```

#### Issue: Training job OOM errors

```python
# Reduce data size or increase machine type
machine_type="n1-highmem-16"  # More memory
```

#### Issue: Slow inference

```python
# Use optimized deployment
predictor.clone_for_deployment()  # Removes unnecessary artifacts
predictor.persist()  # Keeps models in memory
```

### Debug Mode

Enable debug logging:
```bash
export CLOUD_ML_LOG_LEVEL="DEBUG"
```

### Container Testing

Test containers locally:
```bash
# Test trainer
docker run -e AIP_TRAINING_DATA_URI=gs://bucket/train.csv trainer-image

# Test predictor
docker run -p 8501:8501 predictor-image
```

## 🎯 Advanced Usage

### Multimodal Features

Enable text + tabular learning:
```bash
--multimodal="true"
export AIP_TRAINING_DATA_URI="gs://bucket/text_tabular_image_data.csv"
```

### Feature Importance

Calculate and save feature importance:
```bash
--calculate-importance="true"
```

### Model Resumption

Resume from checkpoint:
```bash
--model-import-uri="gs://bucket/previous-model/"
```

### Custom Evaluation Metrics

```bash
--eval-metric="accuracy"  # For classification
--eval-metric="rmse"  # For regression
```

## 📚 Additional Resources

- **AutoGluon Documentation**: <https://auto.gluon.ai/>
- **Vertex AI Training**: <https://cloud.google.com/vertex-ai/docs/training>
- **Vertex AI Prediction**: <https://cloud.google.com/vertex-ai/docs/predictions>
- **AutoGluon Tutorials**: <https://auto.gluon.ai/tutorials/>

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
