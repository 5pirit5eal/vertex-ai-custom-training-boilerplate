import logging
import os
import sys

import torch
from autogluon.tabular import TabularPredictor

from trainer.config import Config, load_config
from trainer.data import convert_gs_to_gcs, load_data, write_df, write_json


def main():
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(message)s",
    )
    logging.getLogger("autogluon").removeHandler(
        logging.getLogger("autogluon").handlers[0]
    )
    logging.getLogger("autogluon").addHandler(logging.StreamHandler(sys.stdout))
    # Load the training data.
    logging.info("Loading config...")
    config: Config = load_config.main(standalone_mode=False)
    logging.info(f"Config loaded: {config}")

    logging.getLogger().setLevel(
        logging.getLevelNamesMapping().get(config.log_level, logging.DEBUG)
    )

    if config.use_gpu:
        logging.info("GPU availability: %s", str(torch.cuda.is_available()))

    # Load the data
    logging.info("Loading data...")
    train_df, val_df, test_df = load_data(config)

    # Create a TabularPredictor.
    if config.model_import_uri is not None:
        logging.info("Importing model...")
        predictor = TabularPredictor.load(config.model_import_uri)
    else:
        predictor = TabularPredictor(
            label=config.label,
            eval_metric=config.eval_metric,
            sample_weight="Weight" if "Weight" in train_df.columns else None,
            path=convert_gs_to_gcs(config.model_export_uri),
            log_to_file=True,
            log_file_path=os.path.join(
                convert_gs_to_gcs(config.tensorboard_log_uri), "training.log"
            ),
        )

    # Fit the model
    logging.info("Fitting model...")
    predictor.fit(
        train_data=train_df,
        tuning_data=val_df,
        presets=config.presets,
        time_limit=config.time_limit,
        num_gpus=1 if config.use_gpu and torch.cuda.is_available() else 0,
        # hyperparameters=training_config.hyperparameters,
        ag_args_ensemble=dict(fold_fitting_strategy="sequential_local"),
    )

    # Predict on the test data
    logging.info("Predicting on test data and writing results...")
    if test_df is not None:
        test_predictions = predictor.predict_proba(test_df)
        # Write the predictions to a CSV file in GCS
        write_df(
            config=config,
            df=test_predictions,
            filenname="test_predictions.csv",
        )

        # Evaulate the model
        test_evaluation = predictor.evaluate_predictions(
            y_true=test_df[config.label],
            y_pred=test_predictions,
            auxiliary_metrics=True,
        )

        # Write the evaluation to a CSV file in GCS
        write_json(
            config=config,
            data=test_evaluation,
            filenname="test_evaluation.json",
        )

        write_df(config, predictor.leaderboard(), "leaderboard.csv")
        if config.calc_importance:
            write_df(
                config,
                predictor.feature_importance(test_df),
                "feature_importance.csv",
            )


if __name__ == "__main__":
    main()
