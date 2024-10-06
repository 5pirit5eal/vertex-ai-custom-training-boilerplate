#!/bin/bash

# Load environment variables from .env
set -o allexport
source .env
set +o allexport

# Create a Google Cloud Storage bucket if it doesn't exist yet
gsutil mb -l $REGION -p $PROJECT_ID gs://$BUCKET_NAME || true

# Create the package
python3 setup.py sdist --format=gztar

# Upload the package
gsutil cp dist/trainer-0.1.tar.gz $BUCKET_URI/trainer-0.1.tar.gz

gsutil ls -l $BUCKET_URI/trainer-0.1.tar.gz

if $1 = "local"; then
    gcloud ai custom-jobs local-run \
        --executor-image-uri="python:3.12" \
        --local-package-path=./trainer \
        --python-module=trainer.task.py \
        --output-image-uri=$TRAIN_DOCKER_IMAGE_URI \
        -- \ # add arguments to be passed to task.py below here
        --job-dir=gs://$BUCKET_NAME/
if $1 = "gcloud"; then
    gcloud ai custom-jobs create \
        --region=$REGION \
        --display-name=$JOB_NAME \
        --config=config.yaml \
