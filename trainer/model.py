from __future__ import unicode_literals

import logging

from xgboost import XGBClassifier

from trainer import metadata


def create(**kwargs) -> XGBClassifier:
    """Create the model by defining a preprocessing transformer
    and a model. Be sure to keep it at 2 steps, as this is necessary
    for correct preprocessing of all data splits.
    """
    logging.info("Creating model...")
    # Create the model with your preprocessing pipeline as part of the model
    # NOTE: This could include one-hot encoding or feature crosses

    model = XGBClassifier(
        **kwargs,
        objective=metadata.OBJECTIVE,
        eval_metric=metadata.EVAL_METRIC,
    )

    return model
