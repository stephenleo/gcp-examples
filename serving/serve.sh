#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")
VERSION_NAME="v2_$now"
MODEL_NAME="name_gender_prediction"
MODEL_DIR="gs://leo-models/gender_prediction/models/2/"

python3 setup.py sdist
gsutil cp dist/name_gender_predictor-2.0.tar.gz gs://leo-models/gender_prediction/package/

gcloud beta ai-platform versions create $VERSION_NAME\
    --model=$MODEL_NAME\
    --origin=$MODEL_DIR\
    --python-version=3.7\
    --runtime-version=1.15\
    --package-uris=gs://leo-models/gender_prediction/package/name_gender_predictor-2.0.tar.gz\
    --prediction-class=predictor.CustomOpTfPredictor