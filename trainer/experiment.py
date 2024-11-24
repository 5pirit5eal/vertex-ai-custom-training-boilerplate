import os
from typing import Literal
import logging

import hypertune
import numpy as np
import xgboost as xgb
from google.auth import default

from xgboost import XGBClassifier

from trainer import model, utils


def evaluate(xgb_model: XGBClassifier):
    logging.info("Getting eval results...")
    eval_results = xgb_model.eval_result()
    logging.info(
        "Eval results for n_estimators %i",
        xgb_model.n_estimators,
        extra={"json_fields": eval_results},
    )

    # report metric for hyperparameter tuning
    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag="accuracy_score",
        metric_value=eval_results["validation_1"]["accuracy_score"][-1],
    )
    return eval_results


def run(
    n_estimators: int = 100,
    learning_rate: float = 2e-5,
    max_depth: int = 6,
    min_child_weight: int = 1,
    gamma: int = 0,
    subsample: float = 1.0,
    lamb: float = 1.0,
    alpha: float = 0.0,
    device: Literal["cpu", "cuda"] = "cpu",
    verbosity: int = 1,
    gs_dir: str = os.getenv("BUCKET_NAME", ""),
    model_name: str = "diabetes-classifier",
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
    # Set up log level
    xgb.set_config(verbosity=verbosity)

    # Convert the gs_dir to google cloud fuse
    gs_dir = utils.convert_gs_to_gcs(gs_dir)

    # Open our dataset
    train_data, train_labels, eval_set = utils.load_data(gs_dir=gs_dir)

    xgb_model = model.create(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        gamma=gamma,
        subsample=subsample,
        lamb=lamb,
        alpha=alpha,
        device=device,
    )
    xgb_model.fit(
        train_data, train_labels, eval_set=[(train_data, train_labels), eval_set]
    )
    evaluate(xgb_model=xgb_model)

    # Save the model to GCS
    utils.save_model(model=xgb_model, gs_dir=gs_dir, model_name=model_name)
