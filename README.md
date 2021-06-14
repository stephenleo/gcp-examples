# gender prediction
## LSTM to predict Gender based on tokopedia names dateset

This repo has been reworked from the ground-up to use GCP AI Platform Training and Model Serving of a tensorflow LSTM model. The model has been trained on over 25M names stored in sharded tfrecords and distributed using tensorflow MultiWorkerMirroredStrategy. Test data accuracy is >90%!

## Testing the latest GCP AI platform published API:
1. Create a JSON file with the names of interest with similar structure as below
    ```
    {"instances":[{"name":"stephen leo"}, {"name":"marie stephen leo"}]}
    ```

2. Update the path to your json file in `testing/online_predict.sh`
    ```
    -d @path_to_your_json_file.json
    ```

3. Run online_predict.sh
    ```
    cd testing
    sh online_predict.sh
    ```

4. Output is a JSON object
    ```
    {'predictions': {'probability': [0.9067956805229187, 0.5227343440055847], 'gender': ['m', 'f']}}
    ```

## Training data is from: 
Table name: `tokopedia-970.voyager_dwh.bi_dim_user`
Data Query: [Notebook](notebooks/00_DataQuery.ipynb)

## Training the model on GCP AI platform:
1. Update the parameters in `training/train.sh`. Especially the model save path
    ```
    -msp="gs://leo-models/gender_prediction/models/x/"
    ```

2. Run train.sh
    ```
    cd training
    sh train.sh
    ```

You can view the progress of your training on [GCP AI Platform Jobs console](https://console.cloud.google.com/ai-platform/jobs?project=toped-ds-sandbox)

## Publishing a Serving API to GCP AI Platform:
1. Update the parameters in `serving/serve.sh`. Especially the `MODEL_DIR`
    ```
    MODEL_DIR="gs://leo-models/gender_prediction/models/2/"
    ```

2. Run serve.sh
    ```
    cd serving
    sh serve.sh
    ```
