import logging
import os
import random
import string
import sys

import pandas as pd
import torch
from autogluon.tabular import TabularPredictor
from google.cloud import aiplatform

from trainer.config import Config, load_config
from trainer.data import (
    gcs_path,
    load_data,
    log_container_execution,
    log_metadata,
    log_nested_metrics,
    log_roc_curve,
    write_df,
    write_instance_and_prediction_schemas,
    write_json,
)


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

    # Add file handler to write logs to a file in the logging directory
    log_path = gcs_path(config.tensorboard_log_uri, "training.log")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(file_handler)
    logging.getLogger("autogluon").addHandler(file_handler)

    if config.use_gpu:
        logging.info("GPU availability: %s", str(torch.cuda.is_available()))

    # Initialize the Vertex AI SDK
    if config.experiment_name:
        logging.info("Initializing Vertex AI SDK...")
        existing_tensorboards = aiplatform.Tensorboard.list(
            filter=f"display_name={config.experiment_name} AND labels.ytrue={config.label}",
            location=config.region,
            project=config.project_id,
        )
        if existing_tensorboards:
            logging.info(
                "Found existing Tensorboard for this experiment, using it: %s",
                existing_tensorboards[0].name,
            )
            tensorboard = aiplatform.Tensorboard(
                existing_tensorboards[0].resource_name
            )
        else:
            logging.info(
                "No existing Tensorboard found for this experiment, creating a new one."
            )
            tensorboard = aiplatform.Tensorboard.create(
                display_name=config.experiment_name,
                location=config.region,
                project=config.project_id,
                labels={"ytrue": config.label},
                is_default=True,
            )

        aiplatform.init(
            project=config.project_id,
            location=config.region,
            experiment=config.experiment_name,
            experiment_tensorboard=tensorboard,
            staging_bucket=gcs_path(config.model_export_uri),
        )
        if config.experiment_run_name:
            aiplatform.start_run(config.experiment_run_name)
        else:
            UUID = generate_uuid()
            aiplatform.start_run(f"autogluon-{config.label}-{UUID}")

        aiplatform.log_params(config.to_dict())

    # Load the data
    logging.info("Loading data...")
    train_df, val_df, test_df = load_data(config)

    if config.weight_column not in train_df.columns:
        logging.info(
            f"Weight column '{config.weight_column}' not found in training data. No weighting will be applied."
        )

    # Create a TabularPredictor.
    if config.model_import_uri is not None:
        logging.info("Importing model from checkpoint...")
        predictor: TabularPredictor = TabularPredictor.load(
            config.model_import_uri
        )
    elif config.checkpoint_uri is not None and os.path.exists(
        gcs_path(config.checkpoint_uri, "fit")
    ):
        predictor = TabularPredictor.load(
            gcs_path(config.checkpoint_uri, "fit")
        )
    else:
        predictor = TabularPredictor(
            label=config.label,
            eval_metric=config.eval_metric,
            sample_weight=config.weight_column
            if config.weight_column in train_df.columns
            else None,  # type: ignore
            path=gcs_path(config.checkpoint_uri, "fit"),
            log_to_file=False,
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
            # hyperparameter_tune_kwargs="auto" if config.hyperparameters else None,
            ag_args_ensemble={
                "fold_fitting_strategy": "sequential_local",
            },  # Required as ray is incompatible with vertex ai custom training
            learning_curves=True,
        )

        log_metadata(
            config=config,
            predictor=predictor,
            prefix="train",
        )

    # Refit the model on the train and validation data
    logging.info("Refitting model on full training data...")
    predictor_refit: TabularPredictor = predictor.clone(
        path=gcs_path(config.checkpoint_uri, "refit_full"),
        return_clone=True,
        dirs_exist_ok=True,
    )

    predictor_refit.refit_full(fit_strategy="auto")

    logging.info("Exporting deployment model for inference...")
    predictor.clone_for_deployment(
        path=gcs_path(config.model_export_uri),
        dirs_exist_ok=True,
    )

    log_metadata(
        config=config,
        predictor=predictor_refit,
        prefix="refit_full",
    )

    # Predict on the test data
    logging.info("Predicting on test data and writing results...")
    if test_df is not None:
        test_predictions: pd.DataFrame = predictor_refit.predict_proba(test_df)  # type: ignore
        # Write the predictions to a CSV file in GCS
        write_df(
            config=config,
            df=test_predictions,
            filename="test_predictions.csv",
        )

        # Evaulate the model
        test_evaluation = predictor_refit.evaluate_predictions(
            y_true=test_df[config.label],
            y_pred=test_predictions,
            display=True,
            auxiliary_metrics=True,
            detailed_report=True,
        )

        confusion_matrix: pd.DataFrame = test_evaluation.pop(
            "confusion_matrix", None
        )

        test_evaluation["confusion_matrix"] = (
            confusion_matrix.to_dict(orient="list")
            if confusion_matrix is not None
            else None
        )

        # Write the evaluation to a CSV file in GCS
        write_json(
            config=config,
            data=test_evaluation,
            filename="test_evaluation.json",
        )
        if config.experiment_name:
            logging.info("Logging evaluation metrics to Vertex AI...")
            classification_report: dict = test_evaluation.pop(
                "classification_report", {}
            )
            log_nested_metrics(classification_report)
            # Remove the confusion matrix from the evaluation metrics
            # as this format is not supported by Vertex AI Experiments
            test_evaluation.pop("confusion_matrix", None)
            aiplatform.log_metrics(test_evaluation)

        write_df(
            config,
            predictor_refit.leaderboard(test_df, refit_full=True, display=True),
            "test_leaderboard.csv",
        )

    if config.experiment_name:
        if predictor_refit.problem_type == "binary":
            logging.info("Logging ROC curve...")
            positive_class = predictor_refit.positive_class
            log_roc_curve(
                label_column=config.label,
                positive_class=positive_class,
                test_df=test_df,
                test_predictions=test_predictions,
            )
        elif predictor_refit.problem_type == "multiclass":
            logging.info("Logging multiclass ROC curve...")
            for class_label in predictor_refit.class_labels:
                log_roc_curve(
                    label_column=config.label,
                    positive_class=class_label,
                    test_df=test_df,
                    test_predictions=test_predictions,
                )

        aiplatform.log_classification_metrics(
            labels=predictor.class_labels,
            matrix=confusion_matrix.to_numpy().tolist(),
            display_name="Confusion Matrix - " + config.label,
        )

        log_container_execution(config)
        write_instance_and_prediction_schemas(
            config=config,
            predictor=predictor_refit,
        )

        # Do feature importance calculation last, as it can be time-consuming
        # and we want to ensure all other metrics are logged first.
        if config.calculate_importance:
            logging.info(
                "Calculating and writing feature importance dataframe..."
            )
            write_df(
                config,
                predictor_refit.feature_importance(
                    test_df,
                    time_limit=0.2 * config.time_limit  # type: ignore
                    if config.time_limit
                    else None,
                ),
                "test_feature_importance.csv",
            )

        logging.info(f"{config.experiment_name} completed successfully.")
        aiplatform.end_run()


if __name__ == "__main__":
    main()
