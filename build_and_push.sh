#!/bin/bash
# Usage: ./build_and_push.sh <trainer|predictor> <PROJECT_ID> <REGION> <ARTIFACT_REGISTRY>
# Example: ./build_and_push.sh trainer vtx-demo europe-west3 my-repo

set -e

COMPONENT=$1
PROJECT_ID=$2
REGION=$3
ARTIFACT_REGISTRY=${4:-docker}

if [ -z "$COMPONENT" ] || [ -z "$PROJECT_ID" ] || [ -z "$REGION" ] || [ -z "$ARTIFACT_REGISTRY" ]; then
  echo "Usage: $0 <trainer|predictor> <PROJECT_ID> <REGION> <ARTIFACT_REGISTRY>"
  exit 1
fi

if [[ "$COMPONENT" != "trainer" && "$COMPONENT" != "predictor" ]]; then
  echo "First argument must be 'trainer' or 'predictor'"
  exit 1
fi

# Add current datetime to image tag (format: YYYYMMDD-HHMMSS)
DATETIME=$(date +"%Y%m%d")

if [ "$COMPONENT" = "trainer" ]; then
  IMAGE_TAG="autogluon-train"
elif [ "$COMPONENT" = "predictor" ]; then
  IMAGE_TAG="autogluon-serve"
fi

IMAGE_URI="$REGION-docker.pkg.dev/$PROJECT_ID/$ARTIFACT_REGISTRY/$IMAGE_TAG"
REPO_URI="$IMAGE_URI:$DATETIME"

DOCKERFILE_PATH="$COMPONENT/Dockerfile"

# Build the Docker image
DOCKER_BUILDKIT=1 docker build --platform linux/amd64 -f "$DOCKERFILE_PATH" "$COMPONENT/." -t "$REPO_URI"

# Push the image to Artifact Registry
docker push "$REPO_URI"

# Tag the image with latest
gcloud artifacts docker tags add "$REPO_URI" "$IMAGE_URI":latest

echo "Image pushed: $REPO_URI"
