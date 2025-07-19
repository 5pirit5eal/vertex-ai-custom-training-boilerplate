"""Main module for the prediction server."""

import logging
import os
import threading

from autogluon.tabular import TabularPredictor
from litestar import Litestar, Request, Response, get, post
from litestar.datastructures import State
from litestar.exceptions import HTTPException
from litestar.logging import LoggingConfig, StructLoggingConfig
from litestar.middleware.logging import LoggingMiddlewareConfig
from litestar.plugins.structlog import StructlogConfig, StructlogPlugin
from litestar.status_codes import (
    HTTP_200_OK,
    HTTP_503_SERVICE_UNAVAILABLE,
    HTTP_500_INTERNAL_SERVER_ERROR,
)
from torch.cuda import is_available

from predictor.prediction import create_prediction
from predictor.schemas import PredictionRequest, PredictionResponse
from predictor.utils import load_model

# Constants
_PORT = int(os.getenv("AIP_HTTP_PORT", 8501))


def startup_model(state: State) -> None:
    """Loads the model in a background thread."""
    logger = logging.getLogger(__name__)
    state.predictor = load_model()
    logger.info(f"Cuda available: {is_available()}")
    # Ensure the predictor is ready for serving
    state.predictor.persist()
    state.is_model_ready = True
    logger.info("Model loaded and ready to serve predictions.")


@get(os.getenv("AIP_HEALTH_ROUTE", "/health"))
async def health_check(state: State) -> Response:
    """Health check endpoint."""
    if not state.is_model_ready:
        return Response(
            content="Model not ready",
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
        )
    return Response(content="Model is ready", status_code=HTTP_200_OK)


@post(os.getenv("AIP_PREDICT_ROUTE", "/predict"))
async def predict(
    data: PredictionRequest, state: State
) -> PredictionResponse | HTTPException:
    """Handles prediction requests."""
    if not state.is_model_ready:
        raise HTTPException(
            detail="Model not ready",
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
        )

    model: TabularPredictor = state.predictor
    try:
        predictions = create_prediction(model, data.instances, data.parameters)
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(
            detail=str(e),
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        )
    return PredictionResponse(predictions=predictions)


def startup(app: Litestar) -> None:
    """Starts the model loading in a background thread."""
    logger = logging.getLogger(__name__)
    app.state.is_model_ready = False
    app.state.predictor = None
    logger.info("Starting model loading in a background thread.")
    thread = threading.Thread(target=startup_model, args=(app.state,))
    thread.start()


def app_exception_handler(request: Request, exc: HTTPException) -> Response:
    return Response(
        content={
            "error": "Prediction failed",
            "path": request.url.path,
            "detail": exc.detail,
        },
        status_code=exc.status_code,
    )


structlog_config = StructlogConfig(
    structlog_logging_config=StructLoggingConfig(
        standard_lib_logging_config=LoggingConfig(
            root={"level": "INFO", "handlers": ["queue_listener"]},
            formatters={
                "standard": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                }
            },
        )
    ),
    middleware_logging_config=LoggingMiddlewareConfig(
        exclude=[os.getenv("AIP_HEALTH_ROUTE", "/health")],
        request_log_fields=("body",),
        response_log_fields=("body",),
    ),
)

structlog_plugin = StructlogPlugin(config=structlog_config)


app = Litestar(
    route_handlers=[health_check, predict],
    on_startup=[startup],
    plugins=[structlog_plugin],
    exception_handlers={HTTPException: app_exception_handler},
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "predictor.main:app",
        host="0.0.0.0",
        port=_PORT,
        reload=False,
        workers=int(os.getenv("N_WORKERS", "4")),
    )
