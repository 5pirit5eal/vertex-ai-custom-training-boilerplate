import logging
import os
from typing import Literal

import hypertune  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]
import xgboost as xgb
from google.auth import default
from google.cloud.logging import Client
from sklearn.metrics import accuracy_score  # type: ignore[import-untyped]

from trainer import model, utils


def evaluate(
    xgb_model: xgb.XGBClassifier,
    validation_set: list[pd.DataFrame | pd.Series],
    test_set: list[pd.DataFrame | pd.Series],
) -> float:
    """Evaluate the model.

    Args:
        xgb_model (XGBClassifier | Pipeline): The model to evaluate.
        validation_set (list[pd.DataFrame | pd.Series]): The validation set with data and labels.
        test_set (list[pd.DataFrame | pd.Series]): The test set with data and labels.
    """
    logging.info("Getting eval results...")
    eval_results = xgb_model.evals_result()

    logging.info(
        "Validation eval results",
        extra={"json_fields": eval_results},
    )

    logging.info("Evaluating model on validation data...")
    y_pred = xgb_model.predict(validation_set[0])
    validation_accuracy = accuracy_score(validation_set[1], (y_pred > 0.5).astype(int))  # type: ignore

    logging.info("Validation accuracy score: %f", validation_accuracy)

    logging.info("Evaluating model on test data...")
    y_pred = xgb_model.predict(test_set[0])
    accuracy = accuracy_score(test_set[1], (y_pred > 0.5).astype(int))  # type: ignore

    logging.info("Test accuracy score: %f", accuracy)

    # report metric for hyperparameter tuning
    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag="accuracy",
        metric_value=validation_accuracy,
    )
    return accuracy  # type: ignore


def run(
    n_estimators: int = 100,
    learning_rate: float = 2e-5,
    max_depth: int = 6,
    min_child_weight: int = 1,
    gamma: int = 0,
    subsample: float = 1.0,
    lamb: float = 1.0,
    alpha: float = 0.0,
    early_stopping_rounds: int = 1,
    device: Literal["cpu", "cuda"] = "cpu",
    verbosity: int = 1,
    gs_dir: str = os.getenv("BUCKET_NAME", ""),
    model_name: str = "diabetes-classifier",
    project_id: str = os.getenv("PROJECT_ID", ""),
) -> None:
    """Load the data, train, evaluate, and export the model for serving and
     evaluating for xgboost.

    Args:
        n_estimators (int): number of estimators to use. Equivalent to number of boosting rounds.
        learning_rate (float): learning rate of the model
        max_depth (int): maximum depth of the tree
        min_child_weight (int): minimum weight for the child node
        gamma (int): L1 regularization term on weights
        subsample (float): subsample ratio of the training data
        lamb (float): L2 regularization term on weights
        alpha (float): L1 regularization term on weights
        objective (str): objective function to use
        eval_metric (str): evaluation metric to use
        device (str): device to use
        verbosity (int): level of verbosity
        gs_dir (str): directory to store the model in
        model_name (str): name of the model
        project_id (str): project id to use
    """
    # Check runtime
    _, found_project_id = default()
    if found_project_id:
        project_id = found_project_id

    os.environ["PROJECT_ID"] = project_id

    client = Client(project=project_id)
    client.setup_logging(
        log_level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper())
    )
    # Set up log level
    xgb.set_config(verbosity=verbosity)

    # Convert the gs_dir to google cloud fuse
    model_id = utils.generate_random_id()

    # Open our dataset
    train_set, eval_set, test_set = utils.load_data(
        gs_dir=gs_dir, model_name=model_name, model_id=model_id
    )

    xgb_model = model.create(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        gamma=gamma,
        subsample=subsample,
        lamb=lamb,
        alpha=alpha,
        early_stopping_rounds=early_stopping_rounds,
        device=device,
    )

    logging.info("Training model %s with ID %s", model_name, model_id)
    xgb_model.fit(
        *train_set,
        eval_set=[train_set, eval_set],
    )
    test_accuracy = evaluate(
        xgb_model=xgb_model, validation_set=eval_set, test_set=test_set
    )

    # Save the model to GCS
    utils.save_model_and_results(
        model=xgb_model,
        accuracy=test_accuracy,
        gs_dir=gs_dir,
        model_name=model_name,
        model_id=model_id,
    )

    logging.info("Finished training model %s with ID %s!", model_name, model_id)
