from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import text_processor
import scipy.spatial.distance

flags = tf.compat.v1.flags

# Configuration
flags.DEFINE_string("data_dir", default="data/",
      help="data directory")
flags.DEFINE_string("graph_dir", default="knowledge_graph/",
      help="directory of graph")
flags.DEFINE_integer("embed_steps", default=100,
      help="number of embedding steps")
flags.DEFINE_integer("graph_size", default=512,
      help="the number of nodes in the graph")

flags.DEFINE_bool("embed", default=True,
      help="whether to embed the graph")
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


def kmeans_input_fn_generator(training, tensors):
    def kmeans_input_fn():
        d = tf.data.Dataset.from_tensors(tensors)
        if training:
            d = d.shuffle(buffer_size=1024)
            d = d.repeat()
        return d

    return kmeans_input_fn


def main(argv=None):
    flags = tf.compat.v1.flags.FLAGS.flag_values_dict()
    for i, key in enumerate(flags.keys()):
        if i > 18:
            print(key + ": " + str(flags[key]))

    cluster_estimator = tf.compat.v1.estimator.experimental.KMeans(model_dir=FLAGS.graph_dir,
                                                                   num_clusters=FLAGS.graph_size,
                                                                   use_mini_batch=False)

    if FLAGS.embed:
        print("***************************************")
        print("Cluster with K-Means")
        print("***************************************")

        train_concepts, train_connections, train_originals = text_processor.relationship_processor("data/glove.6B.300d.txt", "data/relationships.csv")

        train_concepts = np.array(train_concepts)

        # train
        cluster_estimator.train(kmeans_input_fn_generator(True, train_concepts), max_steps=FLAGS.embed_steps)

        # embed the edges
        edges_updates = np.zeros([FLAGS.graph_size, FLAGS.graph_size])
        edges = np.zeros([FLAGS.graph_size, FLAGS.graph_size])
        cluster_indices = list(cluster_estimator.predict_cluster_index(kmeans_input_fn_generator(False, train_concepts)))
        previous_index = -1
        relationship_index = 0
        cluster_centers = cluster_estimator.cluster_centers()
        for i, concept in enumerate(train_concepts):
            if train_connections[i] == []:
                relationship_index += 1
                previous_index = -1
            cluster_index = cluster_indices[i]
            # center = cluster_centers[cluster_index]
            # print("clustered score: " + str(scipy.spatial.distance.euclidean(center, concept)))
            # centers = np.array(cluster_centers)
            # distances = []
            # for i in range(len(centers)):
            #     distances.append(scipy.spatial.distance.euclidean(centers[i], concept))
            # scores = sorted(distances)[:3]
            # print("other scores: " + str(scores))
            if previous_index != -1:
                edges_updates[previous_index, cluster_index] += 1
                edges[previous_index, cluster_index] = 1
            previous_index = cluster_index

            # 'point:', point, 'is in cluster', cluster_index, 'centered at', center

        np.save("GraphEdges", edges)

        print("number of edge updates: " + str(np.sum(edges_updates)))
        print("number of edge: " + str(np.sum(edges)))

        print("Ended Clustering")


if __name__ == '__main__':
    tf.compat.v1.app.run()
