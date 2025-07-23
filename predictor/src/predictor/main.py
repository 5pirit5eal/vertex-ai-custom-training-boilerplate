"""Main module for the prediction server."""

import os
import threading

from autogluon.tabular import TabularPredictor
from litestar import Litestar, MediaType, Request, Response, get, post
from litestar.datastructures import State
from litestar.exceptions import HTTPException
from litestar.logging import LoggingConfig
from litestar.middleware.logging import LoggingMiddlewareConfig
from litestar.status_codes import (
    HTTP_200_OK,
    HTTP_500_INTERNAL_SERVER_ERROR,
    HTTP_503_SERVICE_UNAVAILABLE,
)
from torch.cuda import is_available

from predictor.prediction import create_prediction
from predictor.schemas import (
    PredictionRequest,
    PredictionResponse,
)
from predictor.utils import load_model

# Constants
_PORT = int(os.getenv("AIP_HTTP_PORT", 8501))

# Initialize logging
logging_config = LoggingConfig(
    root={"level": "INFO", "handlers": ["queue_listener"]},
    formatters={
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    },
)

logger = logging_config.configure()("app.main")


def startup_model(state: State) -> None:
    """Loads the model in a background thread."""
    state.predictor = load_model()
    logger.info("Cuda available: %s", is_available())
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
) -> Response[PredictionResponse]:
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
        raise HTTPException(
            detail=str(e),
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        )
    return Response(
        content=PredictionResponse(predictions=predictions),
        media_type=MediaType.JSON,
        status_code=HTTP_200_OK,
    )


def startup(app: Litestar) -> None:
    """Starts the model loading in a background thread."""
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


app = Litestar(
    route_handlers=[health_check, predict],
    on_startup=[startup],
    exception_handlers={HTTPException: app_exception_handler},
    logging_config=logging_config,
    middleware=[LoggingMiddlewareConfig().middleware],
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
