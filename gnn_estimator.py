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

from __future__ import absolute_import, division, print_function, unicode_literals

from graph_nets import blocks
from graph_nets import modules
from graph_nets import utils_tf
import sonnet as snt
import functools
import os


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import text_processor

flags = tf.compat.v1.flags

# Configuration
flags.DEFINE_string("data_dir", default="data/",
      help="data directory")
flags.DEFINE_string("model_dir", default="gnn_model/",
      help="directory of model")
flags.DEFINE_string("graph_dir", default="graph_model/",
      help="directory of graph")
flags.DEFINE_integer("train_steps", default=100000,
      help="number of training steps")
flags.DEFINE_float("dropout", default=0.3,
      help="dropout rate")
flags.DEFINE_integer("global_size", default=300,
      help="the depth of the global state")
flags.DEFINE_integer("seq_len", default=80,
      help="length of the each fact")
flags.DEFINE_integer("batch_size", default=128,
      help="batch size for training")
flags.DEFINE_integer("recurrences", default=5,
      help="number of times graph is processed through gnn")
flags.DEFINE_float("learning_rate", default=1e-5,
      help="learning rate for ADAM optimizer")

flags.DEFINE_bool("train", default=True,
      help="whether to train")
flags.DEFINE_bool("predict", default=True,
      help="whether to predict")
flags.DEFINE_integer("predict_samples", default=10,
      help="the number of samples to predict")
flags.DEFINE_string("description", default="",
      help="description of experiment")

FLAGS = flags.FLAGS
flags = tf.compat.v1.flags.FLAGS.flag_values_dict()
for i, key in enumerate(flags.keys()):
    if i > 18:
        print(key + ": " + str(flags[key]))

SIGNATURE_NAME = "serving_default"
num_choices = 4


def model_fn(features, labels, mode, params):
    sentences = features["input_ids"]
    word_embedding = tf.constant(params['word_embedding'])
    graph_nodes = params['graph_nodes']
    graph_edges = params['graph_edges']
    depth = graph_nodes.shape[1]
    training = mode == tf.estimator.ModeKeys.TRAIN

    # The model is going to send a write signal to the original graph and perform graph computation
    # We will not have a state graph. The purpose of this experiment is to make the simplest model for this task.
    # This is what it will currently do:
    # 1. Find where each word in question will project to in the graph.
    # 2. Add a signal onto each of the nodes to be projected by their words.
    # 3. Perform graph computation for a few rounds.
    # 4. Extract the final global state.
    # 5. Softmax over the different choices and predict.

    padding_mask = tf.cast(tf.not_equal(tf.cast(sentences, tf.int32), tf.constant([[word_embedding.shape[0] - 1]])),
                           tf.int32)  # 0 means the token needs to be masked. 1 means it is not masked.
    padding_mask = tf.reshape(padding_mask, [-1, FLAGS.seq_len, 1])
    sentences = tf.nn.embedding_lookup(word_embedding, sentences)
    sentences = tf.reshape(sentences, [-1, FLAGS.seq_len, 300])
    print("sentences: " + str(sentences))
    print("padding_mask: " + str(padding_mask))
    question_encoder = tf.keras.layers.LSTM(300, dropout=FLAGS.dropout, recurrent_dropout=FLAGS.dropout, return_sequences=True)
    # encoded_question = question_encoder(sentences, training=training)
    encoded_question = sentences
    print("encoded_question: " + str(encoded_question))
    encoded_question = tf.cast(padding_mask, tf.float32) * tf.cast(encoded_question, tf.float32)
    encoded_question = tf.reshape(encoded_question, [-1, 300])

    # The template graph
    globals = np.random.randn(FLAGS.global_size).astype(np.float32)
    nodes = graph_nodes.astype(np.float32)
    edges = np.ones([int(np.sum(graph_edges)), 1]).astype(np.float32)
    senders, receivers = np.nonzero(graph_edges)
    print("senders: " + str(senders))
    print("receivers: " + str(receivers))

    graph_dict = {"globals": globals,
                  "nodes": nodes,
                  "edges": edges,
                  "senders": senders,
                  "receivers": receivers}
    print("encoded_question.shape[0]: " + str(encoded_question.shape[0]))
    batch_of_tensor_data_dicts = [graph_dict for i in range(sentences.shape[0])]
    batch_of_graphs = utils_tf.data_dicts_to_graphs_tuple(batch_of_tensor_data_dicts)
    batch_of_nodes = batch_of_graphs.nodes
    print("batch_of_nodes: " + str(batch_of_nodes))

    # Euclidean distance to identify closest nodes
    na = tf.reduce_sum(tf.square(encoded_question), 1)
    nb = tf.reduce_sum(tf.square(nodes), 1)

    # na as a row and nb as a column vectors
    na = tf.reshape(na, [-1, 1])
    nb = tf.reshape(nb, [1, -1])

    # return pairwise euclidead difference matrix
    distance = tf.sqrt(tf.maximum(na - 2 * tf.matmul(encoded_question, nodes, False, True) + nb, 0.0))
    similarity = 1 - distance

    # calculate attention over the graph
    attention_weights = tf.nn.softmax(similarity, axis=-1)
    closest_nodes = tf.cast(tf.argmin(attention_weights, -1), tf.int32)
    print("closest_nodes: " + str(closest_nodes))

    # # Write the signals onto these nodes
    positions = tf.where(tf.not_equal(tf.reshape(closest_nodes, [-1, FLAGS.seq_len]), 99999))
    print("positions: " + str(positions))
    positions = tf.slice(positions, [0, 0], [-1, 1])  # we only want the first 2 dimensions, since the last dimension is incorrect
    print("positions: " + str(positions))
    positions = tf.cast(positions, tf.int32)
    print("positions: " + str(positions))
    positions = tf.concat([positions, tf.reshape(closest_nodes, [-1, 1])], -1)
    print("positions: " + str(positions))
    # print("compressed: " + str(compressed1))
    # print("norm_duplicate: " + str(tf.reshape(norm_duplicate, [-1, 1])))
    projection_signal = tf.reshape(encoded_question, [-1, 300])
    print("projection_signal: " + str(projection_signal))
    batch_of_nodes = tf.tensor_scatter_nd_add(tf.reshape(batch_of_nodes, [-1, 512, 300]), positions, projection_signal)
    batch_of_nodes = tf.keras.layers.LayerNormalization(epsilon=1e-6)(batch_of_nodes)
    print("batch_of_nodes: " + str(batch_of_nodes))
    batch_of_graphs = batch_of_graphs.replace(nodes=tf.reshape(batch_of_nodes, [-1, 300]))

    def model_fn(size):
        return snt.Sequential([snt.nets.MLP(output_sizes=[size], dropout_rate=FLAGS.dropout)])

    global_model = model_fn(depth)
    edge_model = model_fn(1)
    node_model = model_fn(depth)

    train_graph_network = modules.InteractionNetwork(
        edge_model_fn=lambda: functools.partial(edge_model, is_training=True),
        node_model_fn=lambda: functools.partial(node_model, is_training=True))

    eval_graph_network = modules.InteractionNetwork(
        edge_model_fn=lambda: functools.partial(edge_model, is_training=False),
        node_model_fn=lambda: functools.partial(node_model, is_training=False))

    train_global_block = blocks.GlobalBlock(global_model_fn=lambda: functools.partial(global_model, is_training=True))
    eval_global_block = blocks.GlobalBlock(global_model_fn=lambda: functools.partial(global_model, is_training=False))

    num_recurrent_passes = FLAGS.recurrences
    previous_graphs = batch_of_graphs

    for unused_pass in range(num_recurrent_passes):
        print("previous_graphs" + str(unused_pass) + ": " + str(previous_graphs))
        if training:
            previous_graphs = train_graph_network(previous_graphs)
            previous_graphs = train_global_block(previous_graphs)
        else:
            previous_graphs = eval_graph_network(previous_graphs)
            previous_graphs = eval_global_block(previous_graphs)
    output_graphs = previous_graphs

    output_global = tf.keras.layers.Dropout(FLAGS.dropout)(output_graphs.globals, training=training)
    dense_layer = tf.keras.layers.Dense(1, activation='relu')
    logits = dense_layer(output_global)
    logits = tf.reshape(logits, [-1, num_choices])

    def loss_function(real, pred):
        loss_ = tf.keras.losses.sparse_categorical_crossentropy(real, pred, from_logits=True)
        return tf.reduce_mean(loss_)

    # Calculate the loss
    loss = loss_function(features["answer_id"], logits)

    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(loss, name='loss')
    tf.summary.scalar('loss', loss)

    predictions = {
        'original': features["input_ids"],
        'prediction': tf.argmax(logits, -1),
        'correct': features["answer_id"]
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = {
            SIGNATURE_NAME: tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.compat.v1.train.get_or_create_global_step()

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, beta2=0.98, epsilon=1e-9)

        # Batch norm requires update ops to be added as a dependency to the train_op
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step)
    else:
        train_op = None

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op)

def file_based_input_fn_builder(input_file, sequence_length, batch_size, is_training, drop_remainder):

    name_to_features = {
        "input_ids": tf.io.FixedLenFeature([4, sequence_length], tf.int64),
        "answer_id": tf.io.FixedLenFeature([1], tf.int64)
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.io.parse_single_example(record, name_to_features)

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

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset("question_data/" + input_file + ".tfrecords")
        if is_training:
            d = d.shuffle(buffer_size=1024)
            d = d.repeat()

        d = d.map(lambda record: _decode_record(record, name_to_features)).batch(batch_size=batch_size,
                                                                                 drop_remainder=drop_remainder)

        return d

    return input_fn


def main(argv=None):
    flags = tf.compat.v1.flags.FLAGS.flag_values_dict()
    for i, key in enumerate(flags.keys()):
        if i > 18:
            print(key + ": " + str(flags[key]))

    mirrored_strategy = tf.distribute.MirroredStrategy()
    config = tf.estimator.RunConfig(
        train_distribute=mirrored_strategy, eval_distribute=mirrored_strategy)

    if not os.path.exists("question_data"):
        word_embedding, decoder = text_processor.openbook_question_processor("data/glove.6B.300d.txt", "question_data", FLAGS.seq_len)
        np.save("question_data/word_embedding", word_embedding)
        np.save("question_data/decoder", np.array(decoder))
    else:
        word_embedding = np.load("question_data/word_embedding.npy")
        decoder = list(np.load("question_data/decoder.npy"))
    cluster_estimator = tf.compat.v1.estimator.experimental.KMeans(model_dir="knowledge_graph", num_clusters=512)
    graph_clusters = cluster_estimator.cluster_centers()
    graph_edges = np.load("GraphEdges.npy")

    gnn_estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir,
                                       params={'word_embedding': word_embedding,
                                               'graph_nodes': graph_clusters,
                                               'graph_edges': graph_edges}, config=config)

    train_input_fn = file_based_input_fn_builder(
        input_file="training_questions",
        sequence_length=FLAGS.seq_len,
        batch_size=FLAGS.batch_size,
        is_training=True,
        drop_remainder=True)

    eval_input_fn = file_based_input_fn_builder(
        input_file="validating_questions",
        sequence_length=FLAGS.seq_len,
        batch_size=1,
        is_training=False,
        drop_remainder=True)

    test_input_fn = file_based_input_fn_builder(
        input_file="testing_questions",
        sequence_length=FLAGS.seq_len,
        batch_size=1,
        is_training=False,
        drop_remainder=True)

    if FLAGS.train:
        print("***************************************")
        print("Training")
        print("***************************************")

        trainspec = tf.estimator.TrainSpec(
            input_fn=train_input_fn,
            max_steps=FLAGS.train_steps)

        evalspec = tf.estimator.EvalSpec(
            input_fn=eval_input_fn)

        tf.estimator.train_and_evaluate(gnn_estimator, trainspec, evalspec)

    if FLAGS.predict:
        print("***************************************")
        print("Predicting")
        print("***************************************")

        results = gnn_estimator.predict(input_fn=eval_input_fn, predict_keys=['original', 'prediction', 'correct'])
        total = 0
        correct = 0

        total = 0
        correct = 0

        for i, result in enumerate(results):
            print(i)
            predicted_choice = result['prediction']
            correct_choice = result['correct']
            if i + 1 < FLAGS.predict_samples:
                print("------------------------------------")
                input_question = result['original']
                for choice in input_question:
                    print([decoder[word] for word in choice if word != 1])

                print("predicted_choice: " + str(predicted_choice))
                print("correct_choice: " + str(correct_choice[0]))
            total += 1
            if correct_choice[0] == predicted_choice:
                correct += 1

        print("Accuracy: " + str(correct / total))

def find_similarities(similarity, query_sentence, compare_sentence, tokenizer):
    fig = plt.figure(figsize=(16, 8))
    result = list(range(similarity.shape[1]))

    fontdict = {'fontsize': 10}

    ax = fig.add_subplot(1, 1, 1)

    # This goes on the X-axis
    input_sentence = ['<start>'] + [tokenizer.decode([i]) for i in compare_sentence if i < tokenizer.vocab_size and i != 0] + ['<end>']

    # This goes on the Y-axis
    output_sentence = ['<start>'] + [tokenizer.decode([i]) for i in query_sentence if i < tokenizer.vocab_size and i != 0] + ['<end>']

    ax.set_xticklabels(input_sentence, fontdict=fontdict, rotation=90)

    ax.set_yticklabels(output_sentence, fontdict=fontdict)

    # plot the similarity weights
    ax.matshow(similarity[:len(output_sentence), :len(input_sentence)], cmap='viridis')

    ax.set_xticks(range(len(query_sentence) + 2))
    ax.set_yticks(range(len(result)))

    ax.set_ylim(len(output_sentence) - 1, 0)
    ax.set_xlim(0, len(input_sentence) - 1)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    tf.compat.v1.app.run()
