r"""AutoGluon serving binary.

This module sets up a Litestar web server for serving predictions from a
trained AutoGluon model. The server exposes two endpoints:

1.  `/ping`: A health check endpoint that returns "pong" to
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
import threading

import pandas as pd
from autogluon.tabular import TabularPredictor
from litestar import Litestar, Request, Response, get, post
from litestar.datastructures import State
from litestar.status_codes import (
    HTTP_200_OK,
    HTTP_500_INTERNAL_SERVER_ERROR,
    HTTP_503_SERVICE_UNAVAILABLE,
)
from torch.cuda import is_available

from predictor.utils import download_gcs_dir_to_local

# Constants
_PORT = int(os.getenv("AIP_HTTP_PORT", 8501))
GCS_URI_PREFIX = "gs://"
LOCAL_MODEL_DIR = "/tmp/model/"
OPTIMIZED_MODEL_DIR = "/tmp/opt/"


def load_model(state: State) -> None:
    """Loads the model in a background thread."""
    model_dir = os.getenv("AIP_STORAGE_URI", "/model/")
    logging.info(f"Model directory passed by the user is: {model_dir}")
    if model_dir.startswith(GCS_URI_PREFIX):
        # Expect the model to be in a model/ subdirectory
        if not model_dir.endswith("model/"):
            model_dir = os.path.join(model_dir, "model/")
        gcs_path = model_dir[len(GCS_URI_PREFIX) :]
        local_model_dir = os.path.join(LOCAL_MODEL_DIR, gcs_path)

        if os.path.exists(local_model_dir):
            # Other workers might be downloading the model, wait until it is available
            while not os.path.exists(
                os.path.join(OPTIMIZED_MODEL_DIR, "version.txt")
            ):
                logging.info(f"Waiting until Model is finished downloading...")
                threading.Event().wait(120)
        else:
            logging.info(f"Downloading {model_dir} to {local_model_dir}")
            download_gcs_dir_to_local(model_dir, local_model_dir)
            logging.info(f"Finished downloading model to {local_model_dir}")
            model = TabularPredictor.load(local_model_dir)
            logging.info(
                f"Optimizing model from {local_model_dir} by cloning for deployment."
            )
            model.clone_for_deployment(
                path=OPTIMIZED_MODEL_DIR,
                dirs_exist_ok=True,
            )
    else:
        logging.info(f"Model directory is local: {model_dir}")
        if not os.path.exists(model_dir):
            raise FileNotFoundError(
                f"Model directory {model_dir} does not exist."
            )
        model = TabularPredictor.load(model_dir)
        logging.info(
            f"Optimizing model from {model_dir} by cloning for deployment."
        )
        model.clone_for_deployment(
            path=OPTIMIZED_MODEL_DIR,
            dirs_exist_ok=True,
        )

    logging.info(f"Cuda available: {is_available()}")

    not_loaded = True
    while not_loaded:
        try:
            logging.info(
                f"Trying to load optimized model from {OPTIMIZED_MODEL_DIR}"
            )
            state.predictor = TabularPredictor.load(OPTIMIZED_MODEL_DIR)
            not_loaded = False
        except Exception as e:
            logging.error(f"Failed to load optimized model: {e}")
            logging.info("Retrying in 5 seconds...")
            threading.Event().wait(5)

    # Ensure the predictor is ready for serving
    state.predictor.persist()
    state.is_model_ready = True
    logging.info("Model loaded and ready to serve predictions.")


@get(os.getenv("AIP_HEALTH_ROUTE", "/ping"))
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
        df_to_predict = pd.DataFrame(instances)
        model: TabularPredictor = state.predictor

        if model.problem_type not in ["binary", "multiclass"]:
            predictions: pd.DataFrame = model.predict(df_to_predict)
        else:
            predictions: pd.DataFrame = model.predict_proba(df_to_predict)

        response = {"predictions": predictions.to_dict(orient="records")}
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
