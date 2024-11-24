# Data URL for FYI
DATA_URL = "kaggle datasets download -d iammustafatz/diabetes-prediction-dataset"

# Non-tunable model parameters
OBJECTIVE = "binary:logistic"
EVAL_METRIC = ["error", "aucpr", "logloss"]
