#!/bin/bash
REGION=us-central1
ENDPOINT_NAME=name_gender_prediction
PROJECT_ID=leo-gcp-sanbox
INPUT_DATA_FILE=names.json

ENDPOINT_ID=$(echo $(gcloud beta ai endpoints list \
    --region=$REGION \
    --filter=$ENDPOINT_NAME) | cut -d' ' -f3)

curl \
-X POST \
-H "Authorization: Bearer $(gcloud auth print-access-token)" \
-H "Content-Type: application/json" \
https://us-central1-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/us-central1/endpoints/${ENDPOINT_ID}:predict \
-d "@${INPUT_DATA_FILE}"