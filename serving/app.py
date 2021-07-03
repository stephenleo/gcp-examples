# json input: {"instances":[{"name":"stephen"}, {"name":"stephanie"}]}
# Very good explanation of how to format request to ai-platform
# https://stackoverflow.com/questions/49172710/what-does-google-cloud-ml-engine-do-when-a-json-request-contains-bytes-or-b6

import tensorflow as tf
import os
import logging
from flask import Flask, request

class CustomOpTfPredictor:
    def __init__(self, model_dir):
        # Load the model during the init function to speed up predictions
        self.imported = tf.saved_model.load(model_dir)

        # Load the model signatures
        self.f = self.imported.signatures["serving_default"]
        
    def predict(self, instances, **kwargs):       
        # Input Pre-Process
        x_processed = self.to_tensor_format(instances)

        # Predict
        predictions = tf.map_fn(lambda x:self.f(x)["dense"], x_processed)
        predictions = tf.map_fn(lambda pred: tf.squeeze(pred), predictions)

        # Classes
        class_names = tf.constant(["F", "M"], dtype=tf.string)

        # Predictions are output from sigmoid so float32 in range 0 -> 1
        # Round to integers for predicted class and string lookup for class name
        prediction_integers = tf.cast(tf.math.round(predictions), tf.int32) 
        predicted_classes = tf.map_fn(lambda idx: class_names[idx], prediction_integers, dtype=tf.string)

        # Convert sigmoid output for probability
        # 1 (male) will remain at logit output
        # 0 (female) will be 1.0 - logit to give probability
        def to_probability(logit):
            if logit < 0.5:
                return 1.0 - logit
            else:
                return logit
        class_probability = tf.map_fn(to_probability, predictions, dtype=tf.float32)
        
        predicted_classes = predicted_classes.numpy().tolist()
        class_probability = class_probability.numpy().tolist()

        return {
            "predictions": [
                {
                    "gender": gender.decode("utf-8"), 
                    "probability": class_probability[idx]
                } for idx, gender in enumerate(predicted_classes)
            ]
        }
    
    # Pre processing
    def to_tensor_format(self, input_name):
        # Convert name to number
        input_name = tf.constant([name["name"] for name in input_name])
        x_processed = tf.map_fn(lambda name: self.x_preprocess([name]), input_name, dtype=tf.float32)

        return x_processed

    def dynamic_padding(self, inp, min_size):
        """Pad after the name with spaces to make all names min_size long"""
        # https://stackoverflow.com/questions/42334646/tensorflow-pad-unknown-size-tensor-to-a-specific-size
        pad_size = min_size - tf.shape(inp)[0]
        paddings = [[0, pad_size]] # Pad behind the name with spaces to align with padding from to_tensor default_value
        return tf.pad(inp, paddings, mode="CONSTANT", constant_values=" ")

    def x_preprocess(self, x):
        """Preprocess the names
            1. Lowercase all letters
            2. Split on characters
            3. Pad if necessary with spaces after the names till all names are filter_size long. If longer than filter_size then limit to first filter_size chars
            4. Convert letters to numbers and subtract 96 (UTF value of a -1) to make a=1
            5. Make any <0 or >26 numbers as 0 to remove special characters and space"""

        # 1. Lowercase all letters
        x_processed = tf.strings.lower(x)

        # 2. Split on characters
        x_processed = tf.strings.unicode_split(x_processed, input_encoding="UTF-8").to_tensor(default_value=" ") 
        #TODO: in TF2.0 can add an argument shape=[batch_size,100] to do padding/pruning here
        
        # 3. Pad if necessary with spaces after the names till all names are filter_size long. If longer than filter_size then limit to first filter_size chars
        filter_size=100
        x_processed = tf.cond(tf.less(tf.shape(x_processed)[1], filter_size), 
                            true_fn=lambda: tf.map_fn(lambda inp_name: self.dynamic_padding(inp_name, filter_size), x_processed), 
                            false_fn=lambda: tf.map_fn(lambda inp_name: tf.slice(inp_name, tf.constant([0]), tf.constant([100])), x_processed))
        
        # 4. Convert letters to numbers and subtract 96 (UTF value of a -1) to make a=1
        x_processed = tf.strings.unicode_decode(x_processed, 'UTF-8')-96

        # 5. Make any <0 or >26 numbers as 0 to remove special characters and space
        x_processed = tf.map_fn(lambda item: (tf.map_fn(lambda subitem: 0 if (subitem[0]<0 or subitem[0]>26)else subitem[0], item)), x_processed.to_tensor())

        
        # Unique to predictor.py
        x_processed = tf.cast(x_processed, tf.float32)
        
        return x_processed
    
# Flask App
app = Flask(__name__)
app.logger.setLevel(logging.INFO)

print(os.environ['AIP_STORAGE_URI'])
print(os.environ['AIP_HTTP_PORT'])
print(os.environ['AIP_PREDICT_ROUTE'])
print(os.environ['AIP_HEALTH_ROUTE'])

model_dir = os.environ['AIP_STORAGE_URI']
    
pred_model = CustomOpTfPredictor(model_dir)
    
# Predict Function
@app.route(os.environ['AIP_PREDICT_ROUTE'], methods=['GET', 'POST'])
def predict():
    content = request.get_json()
    app.logger.info(content)
    return pred_model.predict(content['instances'])

# Health Check
@app.route(os.environ['AIP_HEALTH_ROUTE'], methods=['GET', 'POST'])
def health_check():
    return {'success': True, 'status': 200}
    
app.run(host='0.0.0.0', port=os.environ['AIP_HTTP_PORT'])