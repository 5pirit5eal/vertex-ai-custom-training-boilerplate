import argparse
import os

from trainer import experiment


def get_args():
    """Define the task arguments with the default values.

    Returns:
        experiment parameters
    """
    args_parser = argparse.ArgumentParser()

    # Experiment arguments
    args_parser.add_argument(
        "--batch-size",
        help="Batch size for each training and evaluation step.",
        type=int,
        default=16,
    )
    args_parser.add_argument(
        "--num-epochs",
        help="""\
        Maximum number of training data epochs on which to train.
        If both --train-size and --num-epochs are specified,
        --train-steps are: (train-size/train-batch-size) * num-epochs.\
        """,
        default=1,
        type=int,
    )
    args_parser.add_argument(
        "--seed",
        help="Random seed (default: 42)",
        type=int,
        default=42,
    )

    # Estimator arguments
    args_parser.add_argument(
        "--learning-rate",
        help="Learning rate value for the optimizers.",
        default=2e-5,
        type=float,
    )
    args_parser.add_argument(
        "--weight-decay",
        help="""
      The factor by which the learning rate should decay by the end of the
      training.

      decayed_learning_rate =
        learning_rate * decay_rate ^ (global_step / decay_steps)

      If set to 0 (default), then no decay occurs.
      If set to 0.5, then the learning rate should reach 0.5 of its original
          value at the end of the training.
      Note that decay_steps is set to train_steps.
      """,
        default=0.01,
        type=float,
    )

    # Enable hyperparameter
    args_parser.add_argument(
        "--hp-tune",
        default="n",
        help='Enable hyperparameter tuning. Valida values are: "y" - enable, "n" - disable',
    )

    # Saved model arguments
    args_parser.add_argument(
        "--gs-dir",
        default=os.getenv("AIP_MODEL_DIR"),
        help="GCS location to export models",
    )
    args_parser.add_argument(
        "--model-name",
        default="finetuned-bert-classifier",
        help="The name of your saved model",
    )
    args_parser.add_argument(
        "--project-id",
        default=os.getenv("GOOGLE_CLOUD_PROJECT", os.getenv("GCLOUD_PROJECT")),
        help="The GCP project id to fall back to if not configured by the environment.",
    )

    return args_parser.parse_args()


def main():
    """Setup / Start the experiment"""
    args = get_args()
    print(args)
    experiment.run(args)


if __name__ == "__main__":
    main()
