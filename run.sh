#!/bin/bash

# Load environment variables from .env
set -o allexport
source .env
set +o allexport

# Create a Google Cloud Storage bucket if it doesn't exist yet
gsutil mb -l $REGION -p $PROJECT_ID gs://$BUCKET_NAME || true

# Create the package
python -m pip install --upgrade pip setuptools wheel
python setup.py sdist --format=gztar

# Upload the package
gsutil cp dist/trainer-0.1.tar.gz gs://$BUCKET_NAME/trainer-0.1.tar.gz

gsutil ls -l gs://$BUCKET_NAME/trainer-0.1.tar.gz

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