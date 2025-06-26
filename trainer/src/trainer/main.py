import logging
import os
import random
import string
import sys

import pandas as pd
import torch
from autogluon.tabular import TabularPredictor
from google.cloud import aiplatform
from sklearn.metrics import roc_curve

from trainer.config import Config, load_config
from trainer.data import (
    convert_gs_to_gcs,
    load_data,
    write_df,
    write_json,
    log_nested_metrics,
    log_learning_curves,
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
    log_path = os.path.join(
        convert_gs_to_gcs(config.tensorboard_log_uri), "training.log"
    )
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(file_handler)
    logging.getLogger("autogluon").addHandler(file_handler)

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
        logging.info("Importing model from checkpoint...")
        predictor: TabularPredictor = TabularPredictor.load(
            config.model_import_uri
        )
    else:
        predictor = TabularPredictor(
            label=config.label,
            eval_metric=config.eval_metric,
            sample_weight="Weight" if "Weight" in train_df.columns else None,
            path=convert_gs_to_gcs(config.model_export_uri),
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
        ag_args_ensemble=dict(fold_fitting_strategy="sequential_local"),
        learning_curves=True,
    )

    # Predict on the test data
    logging.info("Predicting on test data and writing results...")
    if test_df is not None:
        test_predictions = predictor.predict_proba(test_df)
        # Write the predictions to a CSV file in GCS
        write_df(
            config=config,
            df=test_predictions,
            filename="test_predictions.csv",
        )

        # Evaulate the model
        test_evaluation = predictor.evaluate_predictions(
            y_true=test_df[config.label],
            y_pred=test_predictions,
            display=True,
            auxiliary_metrics=True,
            detailed_report=True,
        )

        confusion_matrix: pd.DataFrame = test_evaluation.pop(
            "confusion_matrix", None
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

        write_df(config, predictor.leaderboard(), "leaderboard.csv")
        if config.calc_importance:
            write_df(
                config,
                predictor.feature_importance(
                    test_df, time_limit=0.2 * config.time_limit
                ),
                "feature_importance.csv",
            )

        summary = predictor.fit_summary(show_plot=False)
        del summary["leaderboard"]
        write_json(config, summary, "fit_summary.json")

        metadata, model_data = predictor.learning_curves()
        write_json(config, data=metadata, filename="metadata.json")

    if config.experiment_name:
        logging.info("Logging metrics to Vertex AI...")

        log_learning_curves(model_data)
        if predictor.problem_type == "binary":
            logging.info("Logging ROC curve...")
            positive_class = predictor.class_labels[-1]
            y_true_numerical = test_df[config.label].apply(
                lambda x: 1 if x == positive_class else 0
            )
            fpr, tpr, threshold = roc_curve(
                y_true_numerical, test_predictions[positive_class]
            )
            logging.info(
                f"ROC Curve - {positive_class}:\n"
                f"FPR: {fpr.tolist()}\n"
                f"TPR: {tpr.tolist()}\n"
                f"Thresholds: {threshold.tolist()}"
            )
            aiplatform.log_classification_metrics(
                fpr=fpr.tolist(),
                tpr=tpr.tolist(),
                threshold=threshold.tolist(),
                display_name=f"ROC Curve - {positive_class}",
            )
        aiplatform.log_classification_metrics(
            labels=predictor.class_labels,
            matrix=confusion_matrix.to_numpy().tolist(),
        )

        aiplatform.end_run()


if __name__ == "__main__":
    main()
