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

import pandas as pd
from autogluon.tabular import TabularPredictor
from litestar import Litestar, Request, Response, get, post
from litestar.status_codes import HTTP_200_OK, HTTP_500_INTERNAL_SERVER_ERROR

from predictor.utils import download_gcs_dir_to_local

# Constants
_PORT = int(os.getenv("AIP_HTTP_PORT", 8501))
GCS_URI_PREFIX = "gs://"
LOCAL_MODEL_DIR = "/tmp/model/"


# Model loading
model_dir = os.getenv("AIP_STORAGE_URI", "/model/")
logging.info(f"Model directory passed by the user is: {model_dir}")
if model_dir.startswith(GCS_URI_PREFIX):
    gcs_path = model_dir[len(GCS_URI_PREFIX) :]
    local_model_dir = os.path.join(LOCAL_MODEL_DIR, gcs_path)
    logging.info(f"Download {model_dir} to {local_model_dir}")
    download_gcs_dir_to_local(model_dir, local_model_dir)
    model_dir = local_model_dir
    logging.info(f"Local model directory is: {model_dir}")

predictor = TabularPredictor.load(model_dir)


@get(os.getenv("AIP_HEALTH_ROUTE", "/ping"))
async def ping() -> Response:
    return Response(content="pong", status_code=HTTP_200_OK)


@post(os.getenv("AIP_PREDICT_ROUTE", "/predict"))
async def predict(request: Request) -> Response:
    try:
        data = await request.json()
        instances = data.get("instances", [])
        df_to_predict = pd.DataFrame(instances)
        predictions = predictor.predict(df_to_predict).tolist()
        response = {"predictions": predictions}
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


app = Litestar(route_handlers=[ping, predict])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("predictor.main:app", host="0.0.0.0", port=_PORT, reload=False)
