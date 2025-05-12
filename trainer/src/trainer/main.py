import sys
import os
import logging
from autogluon.tabular import TabularPredictor

from trainer.config import Config, load_config
from trainer.data import convert_gs_to_gcs, load_data, write_df, write_json


def main():
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(message)s",
    )
    # Load the training data.
    logging.info("Loading config...")
    config: Config = load_config.main(standalone_mode=False)
    logging.info(f"Config loaded: {config}")

    logging.getLogger().setLevel(
        logging.getLevelNamesMapping().get(config.log_level, logging.DEBUG)
    )

    # Load the data
    logging.info("Loading data...")
    train_df, val_df, test_df = load_data(config)

    # Create a TabularPredictor.
    predictor = TabularPredictor(
        label=config.label,
        eval_metric=config.eval_metric,
        path=os.path.join(
            convert_gs_to_gcs(config.model_export_uri), "autogluon"
        ),
        sample_weight="Weight" if "Weight" in train_df.columns else None,
    )

    # Fit the model
    logging.info("Fitting model...")
    predictor.fit(
        train_data=train_df,
        tuning_data=val_df,
        presets=config.presets,
        time_limit=config.time_limit,
        # hyperparameters=training_config.hyperparameters,
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
            split="test",
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
            df=test_evaluation,
            filenname="test_evaluation.csv",
        )

        write_df(config, predictor.leaderboard(), "leaderboard.csv")
        write_df(
            config, predictor.feature_importance(), "feature_importance.csv"
        )


if __name__ == "__main__":
    main()
