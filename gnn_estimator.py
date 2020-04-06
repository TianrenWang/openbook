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

    padding_mask = tf.cast(tf.not_equal(tf.cast(sentences, tf.int32), tf.constant([[1]])),
                           tf.int32)  # 0 means the token needs to be masked. 1 means it is not masked.
    padding_mask = tf.reshape(padding_mask, [-1, FLAGS.seq_len, 1])
    sentences = tf.nn.embedding_lookup(word_embedding, sentences)
    sentences = tf.reshape(sentences, [-1, FLAGS.seq_len, depth])
    sentences = tf.cast(sentences, tf.float32)
    # print("sentences: " + str(sentences))
    # print("padding_mask: " + str(padding_mask))
    question_encoder = tf.keras.layers.LSTM(depth, dropout=FLAGS.dropout, return_sequences=True, return_state=True)
    _, encoded_question, _ = question_encoder(sentences, training=training, mask=padding_mask)
    # encoded_question = tf.cast(padding_mask, tf.float32) * tf.cast(encoded_question, tf.float32)
    encoded_question = tf.reshape(tf.tile(encoded_question, [1, FLAGS.seq_len]), [-1, depth])

    # The template graph
    nodes = graph_nodes.astype(np.float32)
    edges = np.ones([int(np.sum(graph_edges)), 1]).astype(np.float32)
    senders, receivers = np.nonzero(graph_edges)
    globals = np.zeros(FLAGS.global_size).astype(np.float32)

    graph_dict = {"globals": globals,
                  "nodes": nodes,
                  "edges": edges,
                  "senders": senders,
                  "receivers": receivers}
    original_graph = utils_tf.data_dicts_to_graphs_tuple([graph_dict])
    graph_dict["nodes"] = nodes * 0
    # print("encoded_question.shape[0]: " + str(encoded_question.shape[0]))
    batch_of_tensor_data_dicts = [graph_dict for i in range(sentences.shape[0])]

    batch_of_graphs = utils_tf.data_dicts_to_graphs_tuple(batch_of_tensor_data_dicts)
    batch_of_nodes = batch_of_graphs.nodes
    # print("batch_of_nodes: " + str(batch_of_nodes))

    # Euclidean distance to identify closest nodes
    sentences = tf.reshape(sentences, [-1, depth])
    na = tf.reduce_sum(tf.square(sentences), 1)
    nb = tf.reduce_sum(tf.square(nodes), 1)

    # na as a row and nb as a column vectors
    na = tf.reshape(na, [-1, 1])
    nb = tf.reshape(nb, [1, -1])

    # return pairwise euclidead difference matrix
    distance = tf.sqrt(tf.maximum(na - 2 * tf.matmul(sentences, nodes, False, True) + nb, 0.0))

    # calculate attention over the graph
    closest_nodes = tf.cast(tf.argmin(distance, -1), tf.int32)
    # print("closest_nodes: " + str(closest_nodes))

    # # Write the signals onto these nodes
    positions = tf.where(tf.not_equal(tf.reshape(closest_nodes, [-1, FLAGS.seq_len]), 99999))
    # print("positions: " + str(positions))
    positions = tf.slice(positions, [0, 0], [-1, 1])  # we only want the first 2 dimensions, since the last dimension is incorrect
    # print("positions: " + str(positions))
    positions = tf.cast(positions, tf.int32)
    # print("positions: " + str(positions))
    positions = tf.concat([positions, tf.reshape(closest_nodes, [-1, 1])], -1)
    # print("positions: " + str(positions))
    # print("compressed: " + str(compressed1))
    # print("norm_duplicate: " + str(tf.reshape(norm_duplicate, [-1, 1])))
    projection_signal = tf.reshape(encoded_question, [-1, depth])
    # print("projection_signal: " + str(projection_signal))
    batch_of_nodes = tf.tensor_scatter_nd_update(tf.reshape(batch_of_nodes, [-1, 512, depth]), positions, projection_signal)
    # print("batch_of_nodes: " + str(batch_of_nodes))
    batch_of_graphs = batch_of_graphs.replace(nodes=tf.reshape(batch_of_nodes, [-1, depth]))

    global_block = blocks.NodesToGlobalsAggregator(tf.math.unsorted_segment_mean)
    global_dense = tf.keras.layers.Dense(depth, activation='relu')

    num_recurrent_passes = FLAGS.recurrences
    previous_graphs = batch_of_graphs
    original_nodes = tf.reshape(original_graph.nodes, [1, 512, depth])
    dropout = tf.keras.layers.Dropout(FLAGS.dropout)
    layernorm_global = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    layernorm_node = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    new_global = global_block(previous_graphs)
    previous_graphs = previous_graphs.replace(globals=global_dense(new_global))
    previous_graphs = previous_graphs.replace(globals=layernorm_global(previous_graphs.globals))
    initial_global = previous_graphs.globals

    model_fn = snt.nets.MLP(output_sizes=[depth])

    for unused_pass in range(num_recurrent_passes):
        # Update the node features with the function
        updated_nodes = model_fn(previous_graphs.nodes)
        updated_nodes = layernorm_node(updated_nodes)
        temporary_graph = previous_graphs.replace(nodes=updated_nodes)
        graph_sum0 = tf.reduce_sum(tf.reshape(tf.math.abs(temporary_graph.nodes), [-1, 4 * 512 * 300]), -1)

        # Send the node features to the edges that are being sent by that node.
        nodes_at_edges = blocks.broadcast_sender_nodes_to_edges(temporary_graph)
        graph_sum1 = tf.reduce_sum(tf.reshape(tf.math.abs(nodes_at_edges), [-1, 4 * 5551 * 300]), -1)

        temporary_graph = temporary_graph.replace(edges=nodes_at_edges)

        # Aggregate the all of the edges received by every node.
        nodes_with_aggregated_edges = blocks.ReceivedEdgesToNodesAggregator(tf.math.unsorted_segment_mean)(
            temporary_graph)
        graph_sum2 = tf.reduce_sum(tf.reshape(tf.math.abs(nodes_with_aggregated_edges), [-1, 4 * 512 * 300]), -1)
        previous_graphs = previous_graphs.replace(nodes=nodes_with_aggregated_edges)

        current_nodes = previous_graphs.nodes
        current_nodes = tf.reshape(current_nodes, [-1, 512, depth])
        current_nodes = dropout(current_nodes, training=training)
        new_nodes = current_nodes * original_nodes
        previous_graphs = previous_graphs.replace(nodes=tf.reshape(new_nodes, [-1, depth]))
        old_global = previous_graphs.globals
        new_global = global_block(previous_graphs)
        previous_graphs = previous_graphs.replace(globals=global_dense(new_global))
        previous_graphs = previous_graphs.replace(globals=layernorm_global(previous_graphs.globals))

    output_global = tf.keras.layers.Dropout(FLAGS.dropout)(previous_graphs.globals, training=training)
    dense_layer = tf.keras.layers.Dense(1)
    logits = dense_layer(output_global)
    logits = tf.reshape(logits, [-1, num_choices])

    def loss_function(real, pred):
        return tf.nn.sparse_softmax_cross_entropy_with_logits(tf.reshape(real, [-1]), pred)

    # Calculate the loss
    loss = loss_function(features["answer_id"], logits)

    predictions = {
        'original': features["input_ids"],
        'prediction': tf.argmax(logits, -1),
        'correct': features["answer_id"],
        'logits': logits,
        'loss': loss,
        'output_global': tf.reshape(output_global, [-1, 4, 300]),
        'initial_global': tf.reshape(initial_global, [-1, 4, 300]),
        'old_global': tf.reshape(old_global, [-1, 4, 300]),
        'new_global': tf.reshape(new_global, [-1, 4, 300]),
        'graph_sum0': graph_sum0,
        'graph_sum1': graph_sum1,
        'graph_sum2': graph_sum2,
        'closest_nodes': tf.reshape(closest_nodes, [-1, 4, FLAGS.seq_len]),
        'input_id': features["input_ids"],
        'mask': tf.reshape(padding_mask, [-1, 4, FLAGS.seq_len]),
        'encoded_question': tf.reshape(encoded_question, [-1, 4, FLAGS.seq_len, depth])
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
            train_op = optimizer.minimize(tf.reduce_mean(loss), global_step)
    else:
        train_op = None

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=tf.reduce_mean(loss),
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
        batch_size=16,
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

        results = gnn_estimator.predict(
            input_fn=eval_input_fn,
            predict_keys=['original', 'prediction', 'correct', 'logits', 'loss',
                          'output_global', 'initial_global', 'new_global', 'old_global', 'graph_sum0', 'graph_sum1', 'graph_sum2',
                          'closest_nodes', 'mask', 'input_id', 'encoded_question'])
        total = 0
        correct = 0

        for i, result in enumerate(results):
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
            print("Logits: " + str(result['logits']) + "     loss: " + str(result['loss']))
            # print("output_global: " + str(np.mean(result['output_global'], -1)))
            # print("initial_global: " + str(result['initial_global']))
            # print("new_global: " + str(result['new_global']))
            # print("old_global: " + str(result['old_global']))
            # print("graph_sum0: " + str(result['graph_sum0']))
            # print("graph_sum1: " + str(result['graph_sum1']))
            # print("graph_sum2: " + str(result['graph_sum2']))
            print("closest_nodes: " + str(result['closest_nodes']))
            # print("mask: " + str(result['mask']))
            # print("input_id: " + str(result['input_id']))
            # print("encoded_question: " + str(np.sum(np.abs(result['encoded_question']), -1)))

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
