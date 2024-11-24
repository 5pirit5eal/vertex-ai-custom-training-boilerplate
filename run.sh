#!/bin/bash

# Load environment variables from .env
set -o allexport
source .env
set +o allexport

# Create a Google Cloud Storage bucket if it doesn't exist yet
gsutil mb -l $REGION -p $PROJECT_ID gs://$BUCKET_NAME || true

# Upload the data to gcs from kaggle
gsutil cp archive/diabetes_prediction_dataset.csv gs://$BUCKET_NAME/data/

# Create the package
uv sync --reinstall
uv build

# Upload the package
gsutil cp dist/vertex_ai_custom_training_xgboost_boilerplate-0.1.0.tar.gz gs://$BUCKET_NAME/$APP_NAME/vertex_ai_custom_training_xgboost_boilerplate-0.1.0.tar.gz

gsutil ls -l gs://$BUCKET_NAME/$APP_NAME/

if [ $1 = "local" ]; then
    # uv pip freeze > requirements.txt
    gcloud ai custom-jobs local-run \
        --executor-image-uri=$TRAIN_DOCKER_IMAGE_URI \
        --local-package-path=. \
        --python-module=trainer.task \
        --extra-packages=dist/vertex_ai_custom_training_xgboost_boilerplate-0.1.0.tar.gz \
        --output-image-uri="xgboost_$APP_NAME" \
        -- \
        --gs-dir="gs://$BUCKET_NAME/" \
        --model_name=$APP_NAME \
        --project_id=$PROJECT_ID
elif [ $1 = "cloud" ]; then
    gcloud ai custom-jobs create \
        --region=$REGION \
        --display-name="xgboost_cloud_training_$APP_NAME" \
        --config="config.yaml"
fi