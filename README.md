# Training, tuning and deploying a PyTorch text sentiment classification model on Vertex AI

Example on how to use vertex ai custom training for a xgboost model. Based on <https://cloud.google.com/vertex-ai/docs/training/containers-overview#how_training_with_containers_works>.

## Prerequisites

Be sure to have done the following before attempting to use this example:

1. Setup a google cloud account, billing, project and artefact registry docker repository
2. Ensure the correct APIs are enabled (e.g. vertex ai)
3. Install `gcloud` CLI (and python >=3.10)
4. Create Application Default Credentials with `gcloud auth application-default login`
5. Configure google cloud docker authentication `gcloud auth configure-docker`
6. Adapt `.env` and `config.yaml` to your specifics based on the above topics
7. Download the csv data of your choice (e.g. dataset to predict diabetes: `diabetes.csv`)

## Package layout

You can structure your training application in any way you like. However, the [following structure](https://cloud.google.com/vertex-ai/docs/training/create-python-pre-built-container#structure) is commonly used in Vertex AI samples, and having your project organized similarly can make it easier for you to follow the samples. This package layout was adapted to more modern python standards and tries to implement more flexibilty while keeping the structure and package simple.

The following python_package directory structure shows a sample packaging approach.

```text
├── pyproject.toml
├── trainer
│   ├── __init__.py
│   ├── experiment.py
│   ├── metadata.py
│   ├── model.py
│   ├── task.py
│   └── utils.py
├── deploy.sh
├── predict.sh
├── run.sh
└── README.md
```

Root directory contains your `pyproject.toml` file with the dependencies. The repository uses `uv` to manage package building and dependencies.
Inside trainer directory:

- `task.py` - Main application module initializes and parse task arguments (hyperparameters). It also serves as an entry point to the trainer.
- `model.py` - Includes a function to create a model with a sequence classification head from a pre-trained model.
- `experiment.py` - Runs the model training and evaluation experiment, and exports the final model.
- `metadata.py` - Defines the metadata for classification tasks such as predefined model, dataset name and target labels.
- `utils.py` - Includes utility functions such as those used for reading data from Cloud Storage and saving models and metric data to Cloud Storage buckets.

The files setup.cfg and setup.py include the instructions for installing the `trainer` package into the operating environment of the Docker image.

The file `trainer/task.py` is the Python script for executing the custom training job.

The file `run.sh` is the main execution script using gcloud to create the training job.

Note: When referred to the file in the worker pool specification, the file suffix(.py) is dropped and the directory slash is replaced with a dot(trainer.task).

## Installation

To install the package use:

```bash
python -m venv .venv --prompt pytorch-example
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e .
```

Alternatively you can just start the script with:

```bash
bash run.sh local
```

for local deployment, or

```bash
bash run.sh cloud
```

for google cloud deployment.

## Deployment

To deploy the model on Vertex AI:

```bash
bash deploy.sh cloud
```

for google cloud deployment.

```bash
bash deploy.sh local
```

for local deployment.

## Prediction

To predict on Vertex AI:

```bash
bash predict.sh cloud "input text"
```
