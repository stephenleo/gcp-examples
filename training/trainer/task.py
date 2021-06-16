#!/usr/bin/env python
# To connect to tensorboard:
    ## On any GCP VM:
    # tensorboard --logdir="gs://leo-us-name-gender/tensorboard/" --port=9120
    
    ## On local machine
    # gcloud compute ssh jupyter@leo-tf -- -NfL 9120:localhost:9120
    ## Open tensorboard in browser by navigating to localhost:9120

import argparse
import tensorflow as tf
import model
import utils
import os

# Bug in TF for MultiWorkerMirroredStrategy (forgot the link)
TF_CONFIG = os.environ.get('TF_CONFIG')
if TF_CONFIG and '"master"' in TF_CONFIG:
    os.environ['TF_CONFIG'] = TF_CONFIG.replace('"master"', '"chief"')

# Bug in TF distributed need to instantiate the strategy at the beginning of the job (forgot the link)
# strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
strategy = tf.distribute.MirroredStrategy()

def get_args():
    args_parser = argparse.ArgumentParser()

    # Data files arguments
    args_parser.add_argument("-tf", "--train_files", help="GCS or local paths to training data.", required=True)
    args_parser.add_argument("-ef", "--eval_files", help="GCS or local paths to evaluation data.", required=True)
    args_parser.add_argument("-tsf", "--test_files", help="GCS or local paths to test data.", required=True)

    # Training and Evaluation arguments
    args_parser.add_argument("-tbs", "--train_batch_size", help="Training batch size", default=64, type=int)
    args_parser.add_argument("-ntex", "--train_examples", help="Numbe of training examples", default=64*5, type=int)
    args_parser.add_argument("-ne", "--num_evals", help="Number of times to evaluate", default=1, type=int)
    args_parser.add_argument("-neex", "--eval_examples", help="Numbe of evaluation examples", default=1000, type=int)
    args_parser.add_argument("-ntsex", "--test_examples", help="Number of test examples", default=100, type=int)

    # Paths
    args_parser.add_argument("-msp", "--model_save_path", help="GS path to save model", default="gs://leo-us-name-gender/model/x/", type=str)
    args_parser.add_argument("-tbp", "--tensorboard_path", help="GS path to tensorboard", default="gs://leo-us-name-gender/tensorboard/", type=str)
    args_parser.add_argument("-cpp", "--checkpoint_path", help="GS path to checkpoints", default="gs://leo-us-name-gender/tmp/", type=str)

    return args_parser.parse_args()

def main():
    args = get_args()
    steps_per_epoch = args.train_examples // (args.train_batch_size * args.num_evals)

    # Read the data
    # TODO: Improve the checking code
    if ".csv" in args.train_files:
        train_set = utils.csv_reader_dataset(args.train_files, batch_size=args.train_batch_size)
    elif ".tfrecord" in args.train_files:
        train_set = utils.tfrecord_reader_dataset(args.train_files, batch_size=args.train_batch_size)
    
    if ".csv" in args.eval_files:
        eval_set = utils.csv_reader_dataset(args.eval_files, batch_size=args.train_batch_size).take(args.eval_examples)
    elif ".tfrecord" in args.eval_files:
        eval_set = utils.tfrecord_reader_dataset(args.eval_files, batch_size=args.train_batch_size).take(args.eval_examples)

    if ".csv" in args.test_files:
        test_set = utils.csv_reader_dataset(args.test_files, batch_size=args.train_batch_size).take(args.test_examples)
    elif ".tfrecord" in args.test_files:
        test_set = utils.tfrecord_reader_dataset(args.test_files, batch_size=args.train_batch_size).take(args.test_examples)

    # Mirrored Strategy to distribute training
    with strategy.scope():
        lstm_model = model.build_dnn_model()

    # Train
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=args.checkpoint_path, save_weights_only=True),
        tf.keras.callbacks.TensorBoard(log_dir=args.tensorboard_path, profile_batch=0) #TF bug: https://github.com/tensorflow/tensorboard/issues/2084
        ]

    history = lstm_model.fit(train_set, validation_data=eval_set, epochs=args.num_evals, steps_per_epoch=steps_per_epoch, validation_steps=steps_per_epoch, callbacks=callbacks)

    # Save trained model
    tf.saved_model.save(lstm_model, export_dir=args.model_save_path)

if __name__ == '__main__':
    main()