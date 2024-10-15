# Task type can be either 'classification', 'regression', or 'custom'.
# This is based on the target feature in the dataset.
TASK_TYPE = "classification"

# Dataset paths

TRAIN_DATA = "gs://cloud-samples-data/ai-platform-unified/datasets/text/happydb/happydb_train.csv"
TEST_DATA = (
    "gs://cloud-samples-data/ai-platform-unified/datasets/text/happydb/happydb_test.csv"
)

# pre-trained model name
PRETRAINED_MODEL_NAME = "bert-base-cased"

# List of the class values (labels) in a classification dataset.
TARGET_LABELS = {
    "leisure": 0,
    "exercise": 1,
    "enjoy_the_moment": 2,
    "affection": 3,
    "achievement": 4,
    "nature": 5,
    "bonding": 6,
}


# maximum sequence length
MAX_SEQ_LENGTH = 128
