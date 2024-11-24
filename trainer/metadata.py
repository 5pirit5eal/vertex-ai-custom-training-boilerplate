import os
from sklearn.metrics import accuracy_score  # type: ignore[import-untyped]

# Data URL for FYI
DATA_URL = "kaggle datasets download -d iammustafatz/diabetes-prediction-dataset"

if os.getenv("BUCKET_NAME") is None:
    raise ValueError("BUCKET_NAME environment variable is not set")

DATA = "/gcs/" + os.getenv("BUCKET_NAME", "") + "/data/diabetes-prediction-dataset.csv"

# pre-trained model name
PRETRAINED_MODEL_NAME = "bert-base-cased"

# Non-tunable model parameters
OBJECTIVE = ("binary:logistic",)
EVAL_METRIC = ["logloss", "aucpr", accuracy_score]
