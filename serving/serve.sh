#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")
PROJECT="leo-gcp-sandbox"
version="v1"
VERSION_NAME="$version_$now"
REGION="us-central1"
ENDPOINT_NAME="name_gender_prediction"
# MODEL_DIR="gs://leo-us-name-gender/model/1/"

# python3 setup.py sdist
# gsutil cp dist/name_gender_predictor-1.0.tar.gz gs://leo-us-name-gender/serving_code/

# gcloud beta ai-platform versions create $VERSION_NAME\
#     --model=$MODEL_NAME\
#     --origin=$MODEL_DIR\
#     --python-version=3.7\
#     --runtime-version=1.15\
#     --package-uris=gs://leo-us-name-gender/serving_code/name_gender_predictor-1.0.tar.gz\
#     --prediction-class=predictor.CustomOpTfPredictor

# Import the Model
gcloud beta ai models upload \
  --region=$REGION \
  --display-name=$VERSION_NAME \
  --container-image-uri=IMAGE_URI \
  --artifact-uri=PATH_TO_MODEL_ARTIFACT_DIRECTORY
    
# Check for an existing Endpoint
ENDPOINT_ID=$(gcloud beta ai endpoints list \
    --region=$REGION \
    --filter=display_name=$ENDPOINT_NAME \
    --uri | rev | cut -d'/' -f1 | rev)

# If none, then create the Endpoint
if [ -z "$ENDPOINT_ID" ]
then
    gcloud beta ai endpoints create --region=$REGION --display-name=$ENDPOINT_NAME
    
    ENDPOINT_ID=$(gcloud beta ai endpoints list \
    --region=$REGION \
    --filter=display_name=$ENDPOINT_NAME \
    --uri | rev | cut -d'/' -f1 | rev)
    
# Deploy the Model
gcloud beta ai endpoints deploy-model $ENDPOINT_ID\
  --region=$REGION \
  --model=$version \
  --display-name=$VERSION_NAME \
  --machine-type=n1-standard-1 \
  --min-replica-count=1 \
  --max-replica-count=2 \
  --traffic-split=0=100