import logging
import os
import sys

import pandas as pd
import torch
from autogluon.tabular import TabularPredictor
from google.cloud import aiplatform

from trainer.config import Config, load_config
from trainer.data import (
    gcs_path,
    load_data,
    write_df,
    write_json,
)
from trainer.experiment import (
    log_nested_metrics,
    log_roc_curve,
    setup_experiment,
    write_metadata,
)
from trainer.vertex import (
    create_multiclass_vertex_ai_eval,
    create_vertex_ai_eval,
    write_model_schemas,
)


def setup_run() -> Config:
    """Sets up the run by loading the configuration and initializing logging."""
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
        setup_experiment(config)

    return config


def load_predictor(config: Config, columns: list[str]) -> TabularPredictor:
    """Loads or creates a TabularPredictor based on the configuration.

    Args:
        config (Config): The configuration object containing the model URIs.
        columns (list[str]): The list of columns in the training data.

    Returns:
        TabularPredictor: The loaded or newly created TabularPredictor.
    """
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
            eval_metric=config.eval_metric,  # type: ignore
            sample_weight=config.sample_weight
            if (
                config.sample_weight in columns
                or config.sample_weight in ["auto_weight", "balanced_weight"]
            )  # type: ignore
            else None,  # type: ignore
            path=gcs_path(config.checkpoint_uri, "fit"),
            log_to_file=False,
        )

    return predictor


def evaluate_df(
    config: Config,
    df: pd.DataFrame,
    predictor: TabularPredictor,
    prefix: str,
    save_predictions: bool = False,  # To be changed once the predictions are useful
) -> None:
    """Evaluates the model on the given DataFrame and writes the results to GCS.

    Args:
        config (Config): The configuration object containing the model URIs.
        df (pd.DataFrame): The DataFrame to evaluate.
        predictor (TabularPredictor): The trained AutoGluon model.
        prefix (str): The prefix for the output files.
        save_predictions (bool): Whether to save the predictions to GCS.
    """
    predictions: pd.DataFrame = predictor.predict_proba(df)  # type: ignore
    if save_predictions:
        # Write the predictions to a CSV file in GCS
        write_df(
            config=config,
            df=predictions,
            filename=f"{prefix}_predictions.csv",
        )

    logging.info(f"EVALUATING {prefix.upper()} DATA")
    # Evaluate the model
    evaluation = predictor.evaluate_predictions(
        y_true=df[config.label],
        y_pred=predictions,
        display=True,
        auxiliary_metrics=True,
        detailed_report=True,
    )

    confusion_matrix: pd.DataFrame = evaluation.pop("confusion_matrix", None)

    evaluation["confusion_matrix"] = (
        confusion_matrix.to_dict(orient="list")
        if confusion_matrix is not None
        else None
    )

    # Write the evaluation to a CSV file in GCS
    logging.info(f"Writing {prefix} evaluation results to GCS...")
    write_json(
        config=config,
        data=evaluation,
        filename=f"{prefix}_evaluation.json",
    )

    if config.experiment_name:
        logging.info(f"Logging evaluation metrics for {prefix} to Vertex AI...")
        classification_report: dict = evaluation.pop(
            "classification_report", {}
        )
        log_nested_metrics(classification_report, prefix=prefix)
        # Remove the confusion matrix from the evaluation metrics
        # as this format is not supported by Vertex AI Experiments
        evaluation.pop("confusion_matrix", None)
        aiplatform.log_metrics(evaluation)

        if predictor.problem_type == "binary":
            logging.info(f"Logging {prefix} ROC curve...")
            positive_class = predictor.positive_class
            log_roc_curve(
                label_column=config.label,
                positive_class=positive_class,
                df=df,
                predictions=predictions,
                prefix=prefix,
            )
        elif predictor.problem_type == "multiclass":
            logging.info(
                f"Logging {prefix} multiclass ovr ROC curves not implemented yet."
            )
            # TODO: Implement multiclass ROC curve

        aiplatform.log_classification_metrics(
            labels=predictor.class_labels,
            matrix=confusion_matrix.to_numpy().tolist(),
            display_name=f"Confusion Matrix - {prefix.upper()} " + config.label,
        )


def evaluate_test_df(
    config: Config,
    test_df: pd.DataFrame,
    predictor: TabularPredictor,
    df: pd.DataFrame,
) -> None:
    """Evaluates the model on the test DataFrame and writes the results to GCS.

    Args:
        config (Config): The configuration object containing the model URIs.
        test_df (pd.DataFrame): The DataFrame to evaluate.
        predictor (TabularPredictor): The trained AutoGluon model.
        predictor_test (TabularPredictor): The predictor to use for testing.
        df (pd.DataFrame): The DataFrame containing the test data.
    """
    logging.info("Evaluating test data...")
    if config.experiment_name:
        write_df(
            config,
            predictor.leaderboard(
                test_df, refit_full=config.refit_full, display=True
            ),
            "leaderboard.csv",
        )
        # Do feature importance calculation last, as it can be time-consuming
        # and we want to ensure all other metrics are logged first.
    if config.calculate_importance:
        logging.info("Calculating and writing feature importance dataframe...")
        write_df(
            config,
            predictor.feature_importance(
                test_df,
                time_limit=0.2 * config.time_limit  # type: ignore
                if config.time_limit
                else None,
            ),
            "test_feature_importance.csv",
        )

    predictions: pd.DataFrame = predictor.predict_proba(test_df)  # type: ignore
    if predictor.problem_type == "binary":
        logging.info("Creating Vertex AI evaluation...")
        positive_class = predictor.positive_class
        vertex_eval = create_vertex_ai_eval(
            label_column=config.label,
            positive_class=positive_class,
            df=df,
            predictions=predictions,
        )
        write_json(
            config=config,
            data=vertex_eval,
            filename="vertex_ai_evaluation.json",
        )
    elif predictor.problem_type == "multiclass":
        logging.info("Creating Vertex AI evaluation for multiclass...")
        vertex_eval = create_multiclass_vertex_ai_eval(
            label_column=config.label,
            df=df,
            predictions=predictions,
            labels=predictor.class_labels,
        )
        write_json(
            config=config,
            data=vertex_eval,
            filename="vertex_ai_evaluation.json",
        )


def main() -> None:
    """Main function to run the training and evaluation process."""
    config: Config = setup_run()

    # Load the data
    logging.info("Loading data...")
    train_df, val_df, test_df = load_data(config)

    if config.sample_weight not in train_df.columns:
        logging.info(
            f"Weight column '{config.sample_weight}' not found in training data. No weighting will be applied."
        )

    # Create a TabularPredictor.
    predictor = load_predictor(config, train_df.columns.to_list())

    # Fit the model
    logging.info("Fitting model...")
    predictor.fit(
        train_data=train_df,
        tuning_data=val_df,  # type: ignore
        presets=config.presets,
        time_limit=config.time_limit,  # type: ignore
        num_gpus="auto" if config.use_gpu and torch.cuda.is_available() else 0,
        hyperparameters=config.hyperparameters,  # type: ignore
        # hyperparameter_tune_kwargs="auto" if config.hyperparameters else None,
        ag_args_ensemble={
            "fold_fitting_strategy": "sequential_local",
        },  # Required as ray is incompatible with vertex ai custom training
        learning_curves=True,
    )

    write_metadata(
        config=config,
        predictor=predictor,
        prefix="train",
    )

    if config.refit_full:
        # Refit the model on the train and validation data
        logging.info("Refitting model on full training data...")
        predictor_refit: TabularPredictor = predictor.clone(  # type: ignore
            path=gcs_path(config.checkpoint_uri, "refit_full"),
            return_clone=True,
            dirs_exist_ok=True,
        )

        predictor_refit.refit_full(fit_strategy="auto")

        write_metadata(
            config=config,
            predictor=predictor_refit,
            prefix="refit_full",
        )

        predictor_test = predictor_refit
    else:
        predictor_test = predictor

    # Predict on training data
    for prefix, df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        if df is not None:
            logging.info(f"Predicting on {prefix} data and writing results...")
            evaluate_df(config, df, predictor_test, prefix)
        else:
            logging.info(f"No {prefix} data available for evaluation.")

    if test_df is not None:
        evaluate_test_df(config, test_df, predictor_test, df)

    write_model_schemas(
        config=config,
        predictor=predictor_test,
    )
    logging.info("Exporting deployment model for inference...")
    predictor_test.clone_for_deployment(
        path=gcs_path(config.model_export_uri),
        dirs_exist_ok=True,
    )
    if config.experiment_name and not config.resume:
        logging.info(f"{config.experiment_name} completed successfully.")
        aiplatform.end_run()


if __name__ == "__main__":
    main()
