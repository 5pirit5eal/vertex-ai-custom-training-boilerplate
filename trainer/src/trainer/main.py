import logging
import os
import random
import string
import sys

import torch
from autogluon.tabular import TabularPredictor
from google.cloud import aiplatform

from trainer.config import Config, load_config
from trainer.data import convert_gs_to_gcs, load_data, write_df, write_json


# Generate a uuid of length 8
def generate_uuid():
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=8))


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

    print(os.environ)

    if config.use_gpu:
        logging.info("GPU availability: %s", str(torch.cuda.is_available()))

    # Initialize the Vertex AI SDK
    if config.experiment_name:
        logging.info("Initializing Vertex AI SDK...")
        aiplatform.init(
            project=config.project_id,
            location=config.region,
            experiment=config.experiment_name,
            staging_bucket=convert_gs_to_gcs(config.model_export_uri),
        )
        aiplatform.autolog(
            disable=True
        )  # Disable autologging to avoid conflicts with autogluon
        if config.experiment_run_name:
            aiplatform.start_run(config.experiment_run_name)
        else:
            UUID = generate_uuid()
            aiplatform.start_run(f"autogluon-{config.label}-{UUID}")
        aiplatform.log_params(config.to_dict())

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
        hyperparameters=config.hyperparameters,
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
        if config.experiment_name:
            aiplatform.log_metrics(test_evaluation)

        write_df(config, predictor.leaderboard(), "leaderboard.csv")
        if config.calc_importance:
            write_df(
                config,
                predictor.feature_importance(
                    test_df, time_limit=0.2 * config.time_limit
                ),
                "feature_importance.csv",
            )
        write_json(
            config,
            predictor.fit_summary(show_plot=True),
            "fit_summary.json",
        )
    if config.experiment_name:
        logging.info("Logging metrics to Vertex AI...")
        aiplatform.end_run()


if __name__ == "__main__":
    main()
