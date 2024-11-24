import os
import logging
from afire import Fire

from trainer import experiment
from google.cloud.logging import Client


if __name__ == "__main__":
    client = Client()
    client.setup_logging(
        log_level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper())
    )
    Fire(experiment.run)
