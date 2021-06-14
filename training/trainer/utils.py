#!/usr/bin/env python
import tensorflow as tf

# Pre processing
def dynamic_padding(inp, min_size):
    """Pad after the name with spaces to make all names min_size long"""
    # https://stackoverflow.com/questions/42334646/tensorflow-pad-unknown-size-tensor-to-a-specific-size
    pad_size = min_size - tf.shape(inp)[0]
    paddings = [[0, pad_size]] # Pad behind the name with spaces to align with padding from to_tensor default_value
    return tf.pad(inp, paddings, mode="CONSTANT", constant_values=" ")

def x_preprocess(x):
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
                          true_fn=lambda: tf.map_fn(lambda inp_name: dynamic_padding(inp_name, filter_size), x_processed), 
                          false_fn=lambda: tf.map_fn(lambda inp_name: tf.slice(inp_name, tf.constant([0]), tf.constant([100])), x_processed))
    
    # 4. Convert letters to numbers and subtract 96 (UTF value of a -1) to make a=1
    x_processed = tf.strings.unicode_decode(x_processed, 'UTF-8')-96

    # 5. Make any <0 or >26 numbers as 0 to remove special characters and space
    x_processed = tf.map_fn(lambda item: (tf.map_fn(lambda subitem: 0 if (subitem[0]<0 or subitem[0]>26)else subitem[0], item)), x_processed.to_tensor())
    
    return x_processed

def preprocess(x, y):
    """tf.data compatible preprocessing"""
    x_processed = x_preprocess(x)
    y_processed = tf.cast(tf.equal(y, "m"), dtype=tf.int32)

    return x_processed, y_processed

def _parse_function(proto):
    """tfrecord format definiton"""
    keys_to_features = {"name": tf.io.FixedLenFeature([], tf.string),
                        "gender": tf.io.FixedLenFeature([], tf.string)}
    
    # Load batch examples
    parsed_features = tf.io.parse_example(proto, keys_to_features)
    x = parsed_features["name"]
    y =  parsed_features["gender"]
       
    return x, y

# Read data
def csv_reader_dataset(filepaths, repeat=1, n_readers=8, n_threads=tf.data.experimental.AUTOTUNE, shuffle_buffer_size=10000, batch_size=32):
    """Read sharded csv files, preprocess, shuffle and create batches using tf.data API"""
    defs = [tf.constant(["NaN"], dtype=tf.string)]*2

    dataset = tf.data.Dataset\
        .list_files(filepaths)\
        .interleave(lambda filepath: tf.data.experimental.CsvDataset(filepath, defs, select_cols=[0,1], header=True), 
                                 cycle_length=tf.data.experimental.AUTOTUNE, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
        .shuffle(shuffle_buffer_size).repeat()\
        .batch(batch_size, drop_remainder=True)\
        .map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
        .prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset

def tfrecord_reader_dataset(filepaths, repeat=1, n_readers=8, n_threads=tf.data.experimental.AUTOTUNE, shuffle_buffer_size=10000, batch_size=32):
    # https://stackoverflow.com/questions/58014123/how-to-improve-data-input-pipeline-performance
    dataset = tf.data.Dataset\
        .list_files(filepaths)\
        .interleave(tf.data.TFRecordDataset, 
                    cycle_length=tf.data.experimental.AUTOTUNE, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
        .shuffle(shuffle_buffer_size).repeat()\
        .batch(batch_size, drop_remainder=True)\
        .map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
        .map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
        .prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset