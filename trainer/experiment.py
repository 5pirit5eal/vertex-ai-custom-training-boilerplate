import os
import numpy as np
import hypertune

import xgboost as xgb
from xgboost import XGBClassifier

from trainer import model, metadata, utils

from google.auth import default


class HPTuneCallback(TrainerCallback):
    """
    A custom callback class that reports a metric to hypertuner
    at the end of each epoch.
    """

    def __init__(self, metric_tag, metric_value):
        super(HPTuneCallback, self).__init__()
        self.metric_tag = metric_tag
        self.metric_value = metric_value
        self.hpt = hypertune.HyperTune()

    def on_evaluate(self, args, state, control, **kwargs):
        print(f"HP metric {self.metric_tag}={kwargs['metrics'][self.metric_value]}")
        self.hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag=self.metric_tag,
            metric_value=kwargs["metrics"][self.metric_value],
            global_step=state.epoch,
        )


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}


def train(args, model, train_dataset, test_dataset):
    """Create the training loop to load pretrained model and tokenizer and
    start the training process

    Args:
      args: read arguments from the runner to set training hyperparameters
      model: The neural network that you are training
      train_dataset: The training dataset
      test_dataset: The test dataset for evaluation
    """
    # TODO: add training loop

    # add hyperparameter tuning callback to report metrics when enabled
    if args.hp_tune == "y":
        trainer.add_callback(HPTuneCallback("accuracy", "eval_accuracy"))

    # training
    trainer.train()

    return trainer


def run(
    num_round: int = 1,
    learning_rate: float = 2e-5,
    max_depth: int = 6,
    min_child_weight: int = 1,
    gamma: int = 0,
    n_estimators: int = 100,
    subsample: float = 1.0,
    lambda: float = 1.0,
    alpha: float = 0.0,
    objective: str = "binary:logistic",
    eval_metric: str = "error",
    device: str = "cpu",
    verbosity: int = 1,
    hp_tune: str = "n",
    gs_dir: str = os.getenv("AIP_MODEL_DIR"),
    model_name: str = "finetuned-bert-classifier",
    project_id: str = os.getenv("GOOGLE_CLOUD_PROJECT"),
):
    """Load the data, train, evaluate, and export the model for serving and
     evaluating.

    Args:
        batch_size (int): Batch size for each training and evaluation step.
        num_epochs (int): Maximum number of training data epochs.
        seed (int): Random seed.
        learning_rate (float): Learning rate value for the optimizers.
        weight_decay (float): The factor by which the learning rate should decay by the end of the training.
        device (str): CPU or GPU.
        verbosity (int): Verbosity level of printing messages. Valid values are 0 (silent), 1 (warning), 2 (info), 3 (debug).
        hp_tune (str): Enable hyperparameter tuning. Valid values are: "y" - enable, "n" - disable.
        gs_dir (str): GCS location to export models.
        model_name (str): The name of your saved model.
        project_id (str): The GCP project id to fall back to if not configured by the environment.
    """
    xgb.set_config()
    # Check runtime
    _, project_id = default()
    if not project_id:
        project_id = project_id

    print(f"Saving files to GCS after training: {gs_dir}")

    # Open our dataset
    train_dataset, test_dataset = utils.load_data()

    label_list = train_dataset.unique("label")
    num_labels = len(label_list)

    # Create the model, loss function, and optimizer
    text_classifier = model.create(num_labels=num_labels)

    # Train / Test the model
    trainer = train(args, text_classifier, train_dataset, test_dataset)

    metrics = trainer.evaluate(eval_dataset=test_dataset)
    trainer.save_metrics("all", metrics)

    # Export the trained model
    trainer.save_model(os.path.join("/tmp", model_name))

    # Save the model to GCS
    if args.gs_dir:
        utils.save_model(args, project_id)
    else:
        print(f"Saved model files at {os.path.join('/tmp', model_name)}")
        print(
            "To save model files in GCS bucket, "
            "please specify job_dir starting with gs://"
        )
