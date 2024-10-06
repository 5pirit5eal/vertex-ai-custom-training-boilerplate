# Training, tuning and deploying a PyTorch text sentiment classification model on Vertex AI

Example on how to use vertex ai custom training for a pytorch model. Based on <https://cloud.google.com/vertex-ai/docs/training/containers-overview#how_training_with_containers_works> and <https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/training/pytorch-text-sentiment-classification-custom-train-deploy.ipynb>.

### Package layout

You can structure your training application in any way you like. However, the [following structure](https://cloud.google.com/vertex-ai/docs/training/create-python-pre-built-container#structure) is commonly used in Vertex AI samples, and having your project organized similarly can make it easier for you to follow the samples.

The following python_package directory structure shows a sample packaging approach.

```text
├── python_package
│   ├── setup.py
│   └── trainer
│       ├── __init__.py
│       ├── experiment.py
│       ├── metadata.py
│       ├── model.py
│       ├── task.py
│       └── utils.py
└── pytorch-text-sentiment-classification-custom-train-deploy.ipynb    --> This notebook
```

Main project directory contains your setup.py file with the dependencies.
Inside trainer directory:

- `task.py` - Main application module initializes and parse task arguments (hyperparameters). It also serves as an entry point to the trainer.
- `model.py` - Includes a function to create a model with a sequence classification head from a pre-trained model.
- `experiment.py` - Runs the model training and evaluation experiment, and exports the final model.
- `metadata.py` - Defines the metadata for classification tasks such as predefined model, dataset name and target labels.
- `utils.py` - Includes utility functions such as those used for reading data, saving models to Cloud Storage buckets.

The files setup.cfg and setup.py include the instructions for installing the package into the operating environment of the Docker image.

The file trainer/task.py is the Python script for executing the custom training job.

Note: When referred to the file in the worker pool specification, the file suffix(.py) is dropped and the directory slash is replaced with a dot(trainer.task).
