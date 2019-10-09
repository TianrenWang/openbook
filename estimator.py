# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import tensorflow as tf

import transformer_model
import text_processor

tf.logging.set_verbosity(tf.logging.INFO)

INPUT_TENSOR_NAME = "inputs"
SIGNATURE_NAME = "serving_default"
training_file = "training.tfrecords"

HEIGHT = 32
WIDTH = 32
DEPTH = 3
NUM_CLASSES = 10
NUM_DATA_BATCHES = 5
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10000 * NUM_DATA_BATCHES
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
RESNET_SIZE = 32
BATCH_SIZE = 1

# Scale the learning rate linearly with the batch size. When the batch size is
# 128, the learning rate should be 0.05.
_INITIAL_LEARNING_RATE = 0.05 * BATCH_SIZE / 128
_MOMENTUM = 0.9

# We use a weight decay of 0.0002, which performs better than the 0.0001 that
# was originally suggested.
_WEIGHT_DECAY = 2e-4

_BATCHES_PER_EPOCH = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE
tf.enable_eager_execution()


def model_fn(features, labels, mode, params):
    facts = features["input_ids"]

    print("facts: " + str(facts))

    #tokenizer = text_processor.get_tokenizer(params['texts'])
    vocab_size = params['vocab_size'] + 2

    network = transformer_model.TED_generator(vocab_size)

    logits = network(facts, mode == tf.estimator.ModeKeys.TRAIN)
    # predictions = tf.argmax(logits, 2)
    # fake_response = tokenizer.decode([c for c in predictions[0] if c < tokenizer.vocab_size])

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))  # Every element that is NOT padded
        # They will have to deal with run on sentences with this kind of setup
        loss_ = tf.keras.losses.sparse_categorical_crossentropy(real, pred, from_logits=True)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    # Add weight decay to the loss.
    loss = loss_function(facts[:, 1:], logits)

    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(loss, name='loss')
    tf.summary.scalar('loss', loss)

    predictions = {
        'prediction': tf.argmax(logits, 2)
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = {
            SIGNATURE_NAME: tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()

        optimizer = tf.train.AdamOptimizer(learning_rate=1e-5, beta2=0.98, epsilon=1e-9)

        # Batch norm requires update ops to be added as a dependency to the train_op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step)
    else:
        train_op = None

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op)

def file_based_input_fn_builder(input_file, is_training, drop_remainder):

    name_to_features = {
      "input_ids": tf.FixedLenFeature([40], tf.int64),
    }

    tf.logging.info("Input tfrecord file {}".format(input_file))

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, tf.int32)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        if is_training:
            batch_size = 128
        else:
            batch_size = 128

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.shuffle(buffer_size=1024)
            d = d.repeat()
            # d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        # d = d.map(lambda record: _decode_record(record, name_to_features)).batch(batch_size=batch_size,
        #                                                                          drop_remainder=drop_remainder)

        return d

    return input_fn


def train(model_dir, data_dir, train_steps):

    mirrored_strategy = tf.distribute.MirroredStrategy()
    config = tf.estimator.RunConfig(
        train_distribute=mirrored_strategy, eval_distribute=mirrored_strategy)

    # Set up logging for predictions
    tensors_to_log = {"loss": "loss"}

    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=10)

    encoded_facts, vocab_size, tokenizer = text_processor.text_processor(data_dir, training_file)

    # train_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={"facts": encoded_facts[:-500]},
    #     batch_size=64,
    #     num_epochs=None,
    #     shuffle=True)

    train_input_fn = file_based_input_fn_builder(
        input_file=training_file,
        is_training=True,
        drop_remainder=True)

    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, params={'vocab_size': vocab_size},
                                       config=config)

    estimator.train(
        input_fn=train_input_fn,
        steps=train_steps)#,
        #hooks=[logging_hook])

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"facts": encoded_facts[-500:]},
        num_epochs=1,
        shuffle=False)

    print("Evaluation loss: " + str(estimator.evaluate(input_fn=eval_input_fn)))

    pred_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"facts": encoded_facts[-1]},
        num_epochs=1,
        shuffle=False)

    results = estimator.predict(input_fn=pred_input_fn, predict_keys=['prediction'])

    for result in results[0]:
        output_sentence = result['prediction']
        print("result: " + str(output_sentence))
        print("decoded: " + str(tokenizer.decode([i for i in output_sentence if i < tokenizer.vocab_size])))
        print("original: " + str(tokenizer.decode([i for i in encoded_facts[-1] if i < tokenizer.vocab_size])))
        break


def main(model_dir, data_dir, train_steps):
    train(model_dir, data_dir, train_steps)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    # For more information:
    # https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html
    args_parser.add_argument(
        '--data-dir',
        default='/data',
        type=str,
        help='The directory where the CIFAR-10 input data is stored. Default: /opt/ml/input/data/training. This '
             'directory corresponds to the SageMaker channel named \'training\', which was specified when creating '
             'our training job on SageMaker')

    # For more information:
    # https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html
    args_parser.add_argument(
        '--model-dir',
        default='/model',
        type=str,
        help='The directory where the model will be stored. Default: /opt/ml/model. This directory should contain all '
             'final model artifacts as Amazon SageMaker copies all data within this directory as a single object in '
             'compressed tar format.')

    args_parser.add_argument(
        '--train-steps',
        type=int,
        default=100,
        help='The number of steps to use for training.')
    args = args_parser.parse_args()
    main(**vars(args))
