#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")
PROJECT=leo-gcp-sandbox
version=v1
REGION=us-central1
ENDPOINT_NAME=name_gender_prediction
VERSION_NAME=${ENDPOINT_NAME}_${version}
MODEL_DIR=gs://leo-us-name-gender-us-central1/models/1/
PORT=5000

# Build the Docker Image
export PROJECT_ID=$(gcloud config list project --format "value(core.project)")
export REPO_NAME=ml-docker-repo
export IMAGE_NAME=${ENDPOINT_NAME}_serving
export IMAGE_TAG=$version
export IMAGE_URI=us-central1-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}

docker build -f Dockerfile -t ${IMAGE_URI} ./

# Push to Artifact Registry
docker push ${IMAGE_URI}

# Import the Model
gcloud beta ai models upload \
  --region=$REGION \
  --display-name=$VERSION_NAME \
  --container-image-uri=$IMAGE_URI \
  --artifact-uri=$MODEL_DIR \
  --container-ports=$PORT \
  --container-predict-route='/predict'

# Get the imported model's ID
MODEL_ID=$(echo $(gcloud beta ai models list \
    --region=$REGION \
    --filter=display_name=$VERSION_NAME) | cut -d' ' -f3)
    
# Check for an existing Endpoint
ENDPOINT_ID=$(echo $(gcloud beta ai endpoints list \
    --region=$REGION \
    --filter=$ENDPOINT_NAME) | cut -d' ' -f3)

# If none, then create the Endpoint
if [ -z "$ENDPOINT_ID" ]
then
    gcloud beta ai endpoints create --region=$REGION --display-name=$ENDPOINT_NAME
    
    ENDPOINT_ID=$(echo $(gcloud beta ai endpoints list \
        --region=$REGION \
        --filter=$ENDPOINT_NAME) | cut -d' ' -f3)
fi
   
# Deploy the Model
gcloud beta ai endpoints deploy-model $ENDPOINT_ID\
  --region=$REGION \
  --model=$MODEL_ID \
  --display-name=$VERSION_NAME \
  --machine-type=n1-standard-2 \
  --min-replica-count=1 \
  --max-replica-count=2 \
  --traffic-split=0=100 \
  --enable-container-logging
