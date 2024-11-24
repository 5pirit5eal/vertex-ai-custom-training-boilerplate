import logging
from xgboost import XGBClassifier

from trainer import metadata


def create(**kwargs):
    """Create the model by loading a pretrained model or define your own."""
    # Create the model with your preprocessing pipeline as part of the model
    # NOTE: This could include one-hot encoding or feature crosses
    logging.info("Creating model...")
    model = XGBClassifier(
        **kwargs, objective=metadata.OBJECTIVE, eval_metric=metadata.EVAL_METRIC
    )

    return model
