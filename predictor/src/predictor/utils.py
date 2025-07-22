"""Utility functions for the prediction server."""

import sys
import logging
import json
import os
import random
import threading
import warnings
from typing import Any
import time

import structlog
from contextvars import ContextVar
from structlog.typing import Processor
from structlog_gcp import build_gcp_processors
import pandas as pd
from autogluon.tabular import TabularPredictor
from google.cloud import storage
from litestar.connection import ASGIConnection
from litestar.middleware import MiddlewareProtocol
from litestar.types import ASGIApp, Receive, Scope, Send
from litestar.enums import ScopeType

PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")
GCS_URI_PREFIX = "gs://"
LOCAL_MODEL_DIR = "/tmp/model/"

request_start_time_var: ContextVar[float | None] = ContextVar(
    "request_start_time", default=None
)


def download_gcs_dir_to_local(gcs_dir: str, local_dir: str) -> None:
    """Downloads files in a GCS directory to a local directory.

    For example:
      download_gcs_dir_to_local(gs://bucket/foo, /tmp/bar)
      gs://bucket/foo/a -> /tmp/bar/a
      gs://bucket/foo/b/c -> /tmp/bar/b/c

    Args:
      gcs_dir: A string of directory path on GCS.
      local_dir: A string of local directory path.
    """
    logger = structlog.get_logger(__name__)
    if not gcs_dir.startswith("gs://"):
        raise ValueError(f"{gcs_dir} is not a GCS path starting with gs://.")
    bucket_name = gcs_dir.split("/")[2]
    prefix = gcs_dir[len("gs://" + bucket_name) :].strip("/") + "/"
    client = storage.Client(project=PROJECT_ID)
    blobs = client.list_blobs(bucket_name, prefix=prefix)
    for blob in blobs:
        if blob.name[-1] == "/":
            continue
        file_path = blob.name[len(prefix) :].strip("/")
        local_file_path = os.path.join(local_dir, file_path)
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        logger.info(
            "Downloading file",
            file_path=file_path,
            local_file_path=local_file_path,
        )
        blob.download_to_filename(local_file_path)


def parse_instances_to_dataframe(
    instances: list[dict[str, Any] | list[Any]],
    predictor: TabularPredictor,
) -> tuple[pd.DataFrame, bool]:
    """Parses request instances and converts them to a DataFrame for inference.

    This function handles both objects (dict) and arrays (list),
    and ensures the resulting DataFrame has the correct feature columns based on the
    predictor's feature metadata.

    If array is provided, the instance values are expected to be in order of the predictor's feature metadata.

    Args:
        instances: Either a list of objects (dict) or arrays (list)
        predictor: The AutoGluon TabularPredictor with loaded model

    Returns:
        A tuple containing:
        - pd.DataFrame: A DataFrame with the correct feature columns for inference.
        - bool: True if the input was a list of lists, False otherwise.

    Raises:
        ValueError: If instances format is invalid or required features are missing
    """
    # Get the expected feature names from the predictor
    feature_metadata = predictor.feature_metadata_in.to_dict()
    expected_features = list(feature_metadata.keys())

    if not instances:
        return pd.DataFrame(columns=expected_features), False

    first_instance = instances[0]
    is_list = isinstance(first_instance, list)

    if isinstance(first_instance, dict):
        df = pd.DataFrame(instances)
        missing_features = set(expected_features) - set(df.columns)
        if missing_features:
            raise ValueError(
                f"Missing required features: {list(missing_features)}"
            )
        return df[expected_features], is_list

    elif isinstance(first_instance, list):
        num_expected_features = len(expected_features)
        for i, instance in enumerate(instances):
            if len(instance) != num_expected_features:
                raise ValueError(
                    f"Instance {i} has {len(instance)} values, but {num_expected_features} are expected."
                )
        return pd.DataFrame(instances, columns=expected_features), is_list

    raise ValueError(
        "Invalid instances format. Provide a list of JSON objects (dict) or a list of arrays (list)."
    )


def load_model() -> TabularPredictor:
    """Loads the model from the path specified by the AIP_STORAGE_URI.

    This function includes logic to handle multiple workers trying to download the model
    at the same time.
    """
    logger = structlog.get_logger(__name__)
    # Wait the thread for a random few seconds to avoid race conditions
    threading.Event().wait(random.randint(0, 5))

    model_dir = os.getenv("AIP_STORAGE_URI", "/model/")
    logger.info("Model directory passed by user", model_dir=model_dir)

    if model_dir.startswith(GCS_URI_PREFIX):
        gcs_path = model_dir[len(GCS_URI_PREFIX) :]
        local_model_dir = os.path.join(LOCAL_MODEL_DIR, gcs_path)

        if os.path.exists(local_model_dir):
            # Other workers might be downloading the model, wait until it is available
            # version.txt is the last file to be downloaded, so we wait for it
            while not os.path.exists(
                os.path.join(local_model_dir, "version.txt")
            ):
                logger.info("Waiting until Model is finished downloading...")
                threading.Event().wait(15)

            # Ensure the version.txt file is fully loaded before proceeding
            threading.Event().wait(5)
        else:
            logger.info(
                "Downloading model",
                source=model_dir,
                destination=local_model_dir,
            )
            download_gcs_dir_to_local(model_dir, local_model_dir)
            logger.info(
                "Finished downloading model", destination=local_model_dir
            )

        return TabularPredictor.load(local_model_dir)

    else:
        logger.info("Model directory is local", model_dir=model_dir)
        if not os.path.exists(model_dir):
            raise FileNotFoundError(
                f"Model directory {model_dir} does not exist."
            )
        return TabularPredictor.load(model_dir)


def setup_logging(service: str) -> None:
    """Sets up structured logging with GCP processors."""
    gcp_processors: list[Processor] = build_gcp_processors(service=service)
    shared_processors = [
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    structlog.configure(
        processors=[
            # Prepare event dict for `ProcessorFormatter`.
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    formatter = structlog.stdlib.ProcessorFormatter(
        # These processors are applied to all log entries, not just structlog ones
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            *gcp_processors,
            structlog.processors.JSONRenderer(),
        ],
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # Configure the root logger to use our structured formatter
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    # Apply the same configuration to specific loggers to ensure consistency
    for logger_name in [
        "litestar",
        "autogluon",
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
    ]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        logger.addHandler(handler)
        # Prevent propagation to avoid duplicate logs
        logger.propagate = False

    structlog.get_logger().info("StructLogging initialized.")
    # Use structlog for this message too
    structlog.get_logger().info("Logging setup complete.")

    # Configure warnings to use the logging system
    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.handlers.clear()
    warnings_logger.addHandler(handler)
    warnings_logger.setLevel(logging.WARNING)
    warnings_logger.propagate = False


class LoggingMiddleware(MiddlewareProtocol):
    """Log HTTP requests and responses."""

    def __init__(self, app: ASGIApp) -> None:
        """Initialize the middleware with a logger."""
        self.app = app
        self.logger: structlog.stdlib.BoundLogger = structlog.get_logger().bind(
            component="http"
        )

    async def __call__(
        self, scope: Scope, receive: Receive, send: Send
    ) -> None:
        """Handle the ASGI request and log details."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Record start time
        start_time = time.time()
        request_start_time_var.set(start_time)

        # Extract request details
        connection = ASGIConnection(scope=scope)
        method = scope["method"]
        path = scope["path"]
        query_string = scope.get("query_string", b"").decode("utf-8")

        # Skip health check logging to reduce noise
        if path == "/health":
            await self.app(scope, receive, send)
            return

        # Log request start
        self.logger.info(
            "request_started",
            method=method,
            path=path,
            query_string=query_string if query_string else None,
            client_ip=connection.client.host if connection.client else None,
            user_agent=connection.headers.get("user-agent"),
        )

        # Track response status
        status_code = 500  # Default to error if not set
        body = None

        async def send_wrapper(message) -> None:
            nonlocal status_code, body
            if message["type"] == "http.response.start":
                status_code = message.get("status", 500)
            elif message["type"] == "http.response.body":
                body = message.get("body", b"")
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(
                "request_failed",
                exc_info=True,
                method=method,
                path=path,
                duration_ms=round(duration * 1000, 2),
                error_type=type(e).__name__,
                error_message=str(e),
            )
            raise
        else:
            # Log request completion
            duration = time.time() - start_time
            self.logger.info(
                "request_completed",
                jsonPayload=json.loads(body.decode("utf-8")) if body else None,
                method=method,
                path=path,
                status_code=status_code,
                duration_ms=round(duration * 1000, 2),
            )
        finally:
            request_start_time_var.set(None)
