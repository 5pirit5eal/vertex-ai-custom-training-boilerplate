r"""AutoGluon serving binary.

This module sets up a Litestar web server for serving predictions from a
trained AutoGluon model. The server exposes two endpoints:

1.  `/health`: A health check endpoint that returns "Model is ready" to
    indicate that the server is running.
2.  `/predict`: An endpoint that accepts POST requests with JSON content.
    Each request should contain one or more instances for which the
    predictions are desired. The endpoint returns the predictions and
    associated probabilities in a JSON response.

The server expects an environment variable `model_path` that points to
the directory where the AutoGluon model artifacts are
stored. If `model_path` is not provided, it defaults to '/autogluon/models'.
"""

import json
import logging
import os
import random
import threading

import pandas as pd
from autogluon.tabular import TabularPredictor
from litestar import Litestar, Request, Response, get, post
from litestar.datastructures import State
from litestar.status_codes import (
    HTTP_200_OK,
    HTTP_400_BAD_REQUEST,
    HTTP_500_INTERNAL_SERVER_ERROR,
    HTTP_503_SERVICE_UNAVAILABLE,
)
from torch.cuda import is_available

from predictor.utils import (
    download_gcs_dir_to_local,
    parse_instances_to_dataframe,
)

# Constants
_PORT = int(os.getenv("AIP_HTTP_PORT", 8501))
GCS_URI_PREFIX = "gs://"
LOCAL_MODEL_DIR = "/tmp/model/"


def load_model(state: State) -> None:
    """Loads the model in a background thread."""
    # Wait the thread for a random few seconds
    threading.Event().wait(random.randint(0, 5))

    model_dir = os.getenv("AIP_STORAGE_URI", "/model/")
    logging.info(f"Model directory passed by the user is: {model_dir}")
    if model_dir.startswith(GCS_URI_PREFIX):
        gcs_path = model_dir[len(GCS_URI_PREFIX) :]
        local_model_dir = os.path.join(LOCAL_MODEL_DIR, gcs_path)

        if os.path.exists(local_model_dir):
            # Other workers might be downloading the model, wait until it is available
            while not os.path.exists(
                os.path.join(local_model_dir, "version.txt")
            ):
                logging.info("Waiting until Model is finished downloading...")
                threading.Event().wait(120)
        else:
            logging.info(f"Downloading {model_dir} to {local_model_dir}")
            download_gcs_dir_to_local(model_dir, local_model_dir)
            logging.info(f"Finished downloading model to {local_model_dir}")

        state.predictor = TabularPredictor.load(local_model_dir)

    else:
        logging.info(f"Model directory is local: {model_dir}")
        if not os.path.exists(model_dir):
            raise FileNotFoundError(
                f"Model directory {model_dir} does not exist."
            )
        state.predictor = TabularPredictor.load(model_dir)

    logging.info(f"Cuda available: {is_available()}")

    # Ensure the predictor is ready for serving
    state.predictor.persist()
    state.is_model_ready = True
    logging.info("Model loaded and ready to serve predictions.")


@get(os.getenv("AIP_HEALTH_ROUTE", "/health"))
async def ping(state: State) -> Response:
    if not state.is_model_ready:
        return Response(
            content="Model not ready",
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
        )
    return Response(content="Model is ready", status_code=HTTP_200_OK)


@post(os.getenv("AIP_PREDICT_ROUTE", "/predict"))
async def predict(request: Request, state: State) -> Response:
    if not state.is_model_ready:
        return Response(
            content=json.dumps({"error": "Model not ready"}),
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            media_type="application/json",
        )
    try:
        data = await request.json()
        instances = data.get("instances", [])
        model: TabularPredictor = state.predictor
        df_to_predict = parse_instances_to_dataframe(instances, model)

        if model.problem_type not in ["binary", "multiclass"]:
            return Response(
                content=json.dumps({"error": "Unsupported problem type"}),
                status_code=HTTP_400_BAD_REQUEST,
                media_type="application/json",
            )
        else:
            predictions = model.predict_proba(df_to_predict)

        if isinstance(predictions, pd.DataFrame):
            response_data = predictions.to_dict(orient="records")
        else:
            response_data = predictions.tolist()

        response = {"predictions": response_data}
        return Response(
            content=json.dumps(response),
            status_code=HTTP_200_OK,
            media_type="application/json",
        )
    except Exception as e:
        return Response(
            content=json.dumps({"error": str(e)}),
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            media_type="application/json",
        )


def startup(app: Litestar) -> None:
    """Starts the model loading in a background thread."""
    app.state.is_model_ready = False
    app.state.predictor = None
    logging.info("Starting model loading in a background thread.")
    thread = threading.Thread(target=load_model, args=(app.state,))
    thread.start()


app = Litestar(
    route_handlers=[ping, predict],
    on_startup=[startup],
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
