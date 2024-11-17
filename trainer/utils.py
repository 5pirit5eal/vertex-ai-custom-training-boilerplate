import os

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from google.cloud import storage

from trainer import metadata


def preprocess_function(examples):
    tokenizer = AutoTokenizer.from_pretrained(
        metadata.PRETRAINED_MODEL_NAME,
        use_fast=True,
    )

    # Tokenize the texts
    tokenizer_args = (examples["text"],)
    result = tokenizer(
        *tokenizer_args,
        padding="max_length",
        max_length=metadata.MAX_SEQ_LENGTH,
        truncation=True,
    )

    # We can extract this automatically but the unique() method of the dataset
    # is not reporting the label -1 which shows up in the pre-processing
    # hence the additional -1 term in the dictionary

    label_to_id = metadata.TARGET_LABELS

    # Map labels to IDs (not necessary for GLUE tasks)
    if label_to_id is not None and "label" in examples:
        result["label"] = [label_to_id[l] for l in examples["label"]]

    return result


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads the data into two different data loaders.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            train_dataset, test_dataset as dataframes
    """
    # dataset loading repeated here to make this cell idempotent
    # since we are over-writing datasets variable

    df_train = pd.read_csv(metadata.TRAIN_DATA)
    df_test = pd.read_csv(metadata.TEST_DATA)

    dataset = DatasetDict(
        {"train": Dataset.from_pandas(df_train), "test": Dataset.from_pandas(df_test)}
    )

    dataset = dataset.map(preprocess_function, batched=True, load_from_cache_file=True)

    train_dataset, test_dataset = dataset["train"], dataset["test"]

    return train_dataset, test_dataset


def save_model(args, project_id):
    """Saves the model to Google Cloud Storage or local file system

    Args:
      args: contains name for saved model.
    """
    scheme = "gs://"
    if args.gs_dir.startswith(scheme):
        gs_dir = args.gs_dir.split("/")
        bucket_name = gs_dir[2]
        object_prefix = "/".join(gs_dir[3:]).rstrip("/")

        if object_prefix:
            model_path = "{}/{}".format(object_prefix, args.model_name)
        else:
            model_path = "{}".format(args.model_name)

        bucket = storage.Client(project=project_id).bucket(bucket_name)
        local_path = os.path.join("/tmp", args.model_name)
        files = [
            f
            for f in os.listdir(local_path)
            if os.path.isfile(os.path.join(local_path, f))
        ]
        for file in files:
            local_file = os.path.join(local_path, file)
            blob = bucket.blob("/".join([model_path, file]))
            blob.upload_from_filename(local_file)
        print(f"Saved model files in gs://{bucket_name}/{model_path}")
    else:
        print(f"Saved model files at {os.path.join('/tmp', args.model_name)}")
        print(
            "To save model files in GCS bucket, please specify gs_dir starting with gs://"
        )
