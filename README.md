# Vertex AI Custom Training Boilerplate

This repository provides a boilerplate for creating custom training and prediction containers for use with Google Cloud's Vertex AI. It includes separate components for training and prediction, along with scripts to build and push Docker images to Google Artifact Registry.

## Project Structure

- **build_and_push.sh**: A script to build and push Docker images for the `trainer` and `predictor` components to Google Artifact Registry.

- **predictor/**: Contains the code and Dockerfile for the prediction service.

  - **src/predictor/**: Includes the main prediction logic and utility functions.

- **trainer/**: Contains the code and Dockerfile for the training service.

  - **src/trainer/**: Includes the main training logic, configuration, and data handling.

## Components

### Trainer

The `trainer` component is responsible for running custom training jobs on Vertex AI. It includes:

- Configuration files for training.

- Data handling scripts.

- Main training logic.

### Predictor

The `predictor` component is responsible for serving predictions. It includes:

- Main prediction logic.

- Utility functions for preprocessing and postprocessing.

## Usage

1. Modify the `trainer` and `predictor` components as needed for your use case.

2. Use the `build_and_push.sh` script to build and push Docker images:

   ```bash
   ./build_and_push.sh <trainer|predictor> <PROJECT_ID> <REGION> <ARTIFACT_REGISTRY>
   ```

   Replace `<PROJECT_ID>`, `<REGION>`, and `<ARTIFACT_REGISTRY>` with your Google Cloud project details.

## Notes

- The `trainer` component's README provides additional details about how Vertex AI mounts Cloud Storage buckets during training.
