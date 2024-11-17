#!/bin/bash

# Load environment variables from .env
set -o allexport
source .env
set +o allexport

# Create a Google Cloud Storage bucket if it doesn't exist yet
gsutil mb -l $REGION -p $PROJECT_ID gs://$BUCKET_NAME || true

# Upload the data to gcs from kaggle
uvx kaggle datasets download -d iammustafatz/diabetes-prediction-dataset -f diabetes.csv
gsutil cp diabetes.csv gs://$BUCKET_NAME

# Create the package
uv build

# Upload the package
gsutil cp dist/vertex_ai_custom_training_xgboost_boilerplate-0.1.0.tar.gz gs://$BUCKET_NAME/vertex_ai_custom_training_xgboost_boilerplate-0.1.0.tar.gz

gsutil ls -l gs://$BUCKET_NAME/vertex_ai_custom_training_xgboost_boilerplate-0.1.0.tar.gz

if [ $1 = "local" ]; then
    gcloud ai custom-jobs local-run \
        --executor-image-uri="europe-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13.py310:latest" \
        --local-package-path=. \
        --python-module=trainer.task \
        --output-image-uri="pytorch_training_$APP_NAME" \
        -- \
        --gs-dir="gs://$BUCKET_NAME/python-example/" \
        --project-id=$PROJECT_ID
elif [ $1 = "cloud" ]; then
    gcloud ai custom-jobs create \
        --region=$REGION \
        --display-name="pytorch_cloud_training_$APP_NAME" \
        --config="config.yaml"
fi