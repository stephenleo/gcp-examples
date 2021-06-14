#!/bin/bash
curl -X POST \
-H "Authorization: Bearer "$(gcloud auth application-default print-access-token) \
-H "Content-Type: application/json; charset=utf-8" \
-d @../data/names_100.json \
https://ml.googleapis.com/v1/projects/toped-ds-sandbox/models/name_gender_prediction:predict

## URL for specific version:
# https://ml.googleapis.com/v1/projects/toped-ds-sandbox/models/name_gender_prediction/versions/v2_20200506_081737:predict