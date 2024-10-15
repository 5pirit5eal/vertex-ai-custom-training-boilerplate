
#!/bin/bash

# Load environment variables from .env
set -o allexport
source .env
set +o allexport

# check that ./predictor/pytorch-cloud-example does not exist
if [ -d "./predictor/pytorch-cloud-example" ]; then
    gsutil -m cp -r gs://$BUCKET_NAME/python-example/pytorch-cloud-example ./predictor/
fi

if [ $1 = "local" ]; then
    docker build -t $DEPLOY_DOCKER_IMAGE_URI .
    docker run -p $SERVING_CONTAINER_PORT:8080 $DEPLOY_DOCKER_IMAGE_URI
elif [ $1 = "cloud" ]; then
    gcloud builds submit --region=$REGION --tag=$DEPLOY_DOCKER_IMAGE_URI .
    gcloud run deploy $APP_NAME --region=$REGION --image=$DEPLOY_DOCKER_IMAGE_URI --cpu=8 --memory=16Gi \
        --allow-unauthenticated \
        --project $PROJECT_ID \
        --max-instances 1
        # --gpu=1 --no-cpu-throttling --gpu-type=nvidia-l4 \
fi
