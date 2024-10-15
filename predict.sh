#!/bin/bash

# Load environment variables from .env
set -o allexport
source .env
set +o allexport

if [ $1 = "local" ]; then
    curl -v "http://localhost:$SERVING_CONTAINER_PORT/$PREDICT_ROUTE" -d "data=$2"
elif [ $1 = "cloud" ]; then
    SERVICE_URL=$(gcloud run services describe $APP_NAME --region $REGION --format 'value(status.url)')
    curl "$SERVICE_URL/$PREDICT_ROUTE" -d "data=$2"
fi