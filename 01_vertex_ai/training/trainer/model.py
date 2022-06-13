#!/usr/bin/env python
import string
import tensorflow as tf

LETTERS = f" {string.ascii_lowercase}"
EMBEDDING = 256

# Model
def build_dnn_model():
    """Construct model using tf.keras API"""
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(len(LETTERS), EMBEDDING, input_length=100))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=128)))#, recurrent_dropout=0.2, dropout=0.2))) #TF Bug causing model saving error if dropout is used. https://github.com/tensorflow/tensorflow/issues/33247
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))  
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
    return model
