#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")
NAME="name_gender_prediction"
TAG="v1"
JOB_NAME="leo_gp_train_$now"
REGION="us-central1"

# Build the Docker Image
export PROJECT_ID=$(gcloud config list project --format "value(core.project)")
export REPO_NAME=ml-docker-repo
export IMAGE_NAME=${NAME}_training
export IMAGE_TAG=$TAG
export IMAGE_URI=us-central1-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}

docker build -f Dockerfile -t ${IMAGE_URI} ./

# Push to Artifact Registry
docker push ${IMAGE_URI}

# Prepare the config file
sed 's|{IMAGE_URI}|'$IMAGE_URI'|g' config_base.yaml > config.yaml

# Start the training
gcloud beta ai custom-jobs create \
    --region=$REGION \
    --display-name=$JOB_NAME \
    --config=config.yaml