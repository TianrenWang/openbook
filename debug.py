import tensorflow as tf
import numpy as np


def normalize_unique(x):
    print("normalize_unique: " + str(x))
    with tf.device('/cpu:0'):
        ___, idx, count = tf.unique_with_counts(x)
    counts = tf.gather(count, idx)
    print("counts: " + str(tf.cast(1 / counts, tf.float32)))
    return tf.cast(1 / counts, tf.float32)


# attention: the attention weights
# k: the number of positions to keep

# returns: values and their corresponding indices that can be used in a gather/scatter operation

def sparsify(attention, k):
    top_values, top_indices = tf.math.top_k(attention, k)
    positions = tf.where(tf.not_equal(top_indices, 99999))
    top_indices = tf.reshape(top_indices, [tf.size(top_indices), 1])
    positions = tf.slice(positions, [0, 0], [-1, len(attention.get_shape().as_list()) - 1])
    positions = tf.cast(positions, tf.int32)
    actual_indices = tf.concat([positions, top_indices], -1)
    top_values = tf.reshape(top_values, [tf.size(top_values)])
    return top_values, actual_indices

if __name__ == '__main__':

    graph_size = 10
    d_model = 1
    batch = 3
    seq = 4
    training = True
    alpha = 0.98
    x = tf.constant(np.random.randn(batch, seq, d_model))

    graphNodes = tf.Variable(tf.constant(np.random.randn(graph_size, d_model), tf.float32), trainable=False)
    compressed = tf.constant(np.random.randn(batch, seq, d_model), tf.float32)
    projection = tf.keras.layers.Dense(d_model)

    # Find the nodes in the graph that are the closest to the encoded signal and update them
    normed_graph = tf.keras.backend.l2_normalize(graphNodes, -1)
    normed_compressed = tf.math.l2_normalize(compressed, -1)
    print("normed_graph: " + str(normed_graph.shape))
    print("normed_compressed: " + str(normed_compressed.shape))

    cosine_similarity = tf.matmul(tf.reshape(normed_compressed, [-1, d_model]),
                                  tf.keras.backend.permute_dimensions(normed_graph, (1, 0)))
    print("cosine_similarity: " + str(cosine_similarity.shape))

    closest_words_ind = tf.cast(tf.argmax(cosine_similarity, -1), tf.int32)  # shape [batch_size * sparse_len], type int64
    print("closest_words_ind: " + str(closest_words_ind.shape))

    print("compressed: " + str(compressed))
    print("graphNodes: " + str(graphNodes))
    oldGraph = graphNodes.value()

    # This part is for training, update the graph node embedding
    if training:
        print("**************Updating Graph Nodes*********************")
        with tf.device('/cpu:0'):
            ___, idx, count = tf.unique_with_counts(closest_words_ind)
        counts = tf.gather(count, idx)
        counts = tf.reshape(tf.cast(counts, tf.float32), [-1, 1])
        closest_words = tf.gather(graphNodes, closest_words_ind) * alpha
        compressed = tf.reshape(compressed, [-1, d_model])
        compressed = compressed * (1 - alpha) / counts
        tf.compat.v1.scatter_nd_update(graphNodes, tf.reshape(closest_words_ind, [-1, 1]), closest_words)

        # tf.compat.v1.scatter_nd_update doesn't accumulate the duplicate updates, so a separate add step is needed
        # print("Update: " + str(tf.compat.v1.scatter_nd_add(graphNodes, tf.reshape(closest_words_ind, [-1, 1]), compressed)))
        tf.compat.v1.scatter_nd_add(graphNodes, tf.reshape(closest_words_ind, [-1, 1]), compressed)

    # This tensor will later be used to visualize which nodes were chosen
    projection_attention = tf.scatter_nd(tf.reshape(closest_words_ind, [-1, 1]),
                                         tf.ones([tf.size(closest_words_ind), 1]),
                                         [graph_size, 1])
    print("graphNodes: " + str(graphNodes))
    print("Difference: " + str(tf.not_equal(oldGraph, graphNodes)))

    # Project signal to the same nodes for added expressiveness
    closest_words_ind_batched = tf.reshape(closest_words_ind, [-1, seq])  # Need to turn it into [batch, graph_len] so that map_fn can work on each sample
    norm_duplicate = tf.expand_dims(tf.map_fn(normalize_unique, closest_words_ind_batched, dtype=tf.float32), -1)
    print("norm_duplicate: "  + str(norm_duplicate.shape))
    batched_nodes = tf.reshape(tf.tile(graphNodes, [tf.shape(x)[0], 1]), [-1] + graphNodes.get_shape().as_list())
    print("batched_nodes: " + str(batched_nodes.shape))
    print("closest_words_ind_batched: " + str(closest_words_ind_batched.shape))
    positions = tf.where(tf.not_equal(closest_words_ind_batched, 99999))
    positions = tf.slice(positions, [0, 0], [-1, 1])  # we only want the first 2 dimensions, since the last dimension is incorrect
    positions = tf.cast(positions, tf.int32)
    positions = tf.concat([positions, tf.reshape(closest_words_ind, [-1, 1])], -1)
    print("compressed: " + str(compressed.shape))
    print("norm_duplicate: " + str(tf.reshape(norm_duplicate, [-1, 1]).shape))
    projection_signal = tf.reshape(projection(compressed), [-1, d_model]) * tf.reshape(norm_duplicate, [-1, 1])
    print("projection_signal: " + str(projection_signal.shape))
    print("positions: " + str(positions))

    encodedGraph = tf.tensor_scatter_nd_add(batched_nodes, positions, projection_signal) # [batch_size, graph_size, FLAGS.d_model]
    print("encodedGraph: " + str(encodedGraph))