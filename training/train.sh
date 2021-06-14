#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="leo_gp_train_$now"
STAGING="gs://leo-models"
REGION="us-west1"

python3 setup.py sdist

gcloud ai-platform jobs submit training $JOB_NAME \
    --staging-bucket $STAGING \
    --region $REGION \
    --scale-tier CUSTOM \
    --master-machine-type n1-standard-8 \
    --master-accelerator count=2,type=nvidia-tesla-k80 \
    --worker-count 3 \
    --worker-machine-type n1-standard-8 \
    --worker-accelerator count=2,type=nvidia-tesla-k80 \
    --runtime-version 1.15 \
    --python-version 3.7 \
    --module-name trainer.task \
    --package-path "/home/jupyter/data-science/official/gender_prediction/training/trainer" \
    -- \
    -tf "gs://leo-models/gender_prediction/data/nonzip/toko_names_train*.tfrecord" \
    -ef "gs://leo-models/gender_prediction/data/nonzip/toko_names_val*.tfrecord" \
    -tsf "gs://leo-models/gender_prediction/data/nonzip/toko_names_test*.tfrecord" \
    -tbs=512 \
    -ntex=25000000 \
    -ne=100 \
    -neex=250000 \
    -msp="gs://leo-models/gender_prediction/models/x/"


    #--scale-tier BASIC_GPU \ # Single machine, single GPU to test out.