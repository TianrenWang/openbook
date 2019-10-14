import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from sklearn.utils import shuffle
from random import shuffle
import os


def create_int_feature(values):
    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return f


def text_processor(data_path, seq_len, processed_path):
    filepath = data_path

    data = open(filepath, "r")
    line = data.readline()
    facts = []
    longest = len(line)
    longest_index = 0
    current_index = 0

    while line:
        first_period_index = line.find('.')
        line = line[:first_period_index]
        facts.append(str.encode(line))
        line = data.readline()
        current_index += 1
        if len(line) > longest:
            longest = len(line)
            longest_index = current_index

    longest_text = facts[longest_index]

    data.close()
    shuffle(facts)

    tokenizer = get_tokenizer(facts)

    # example = tokenizer.encode("An example of camouflage is when something changes color in order to have the same color as its environment")

    # for token in example:
    #     print(tokenizer.decode([token]))

    vocab_size = tokenizer.vocab_size + 2

    def encode(fact):
        """Turns an abstract in English into BPE (Byte Pair Encoding).
        Adds start and end token to the abstract.

        Keyword arguments:
        abstract -- the abstract (type: bytes)
        """

        encoded_fact = [tokenizer.vocab_size] + tokenizer.encode(fact) + [tokenizer.vocab_size + 1]

        return encoded_fact

    # lengths = [0] * 200

    if not os.path.exists(processed_path):
        os.makedirs(processed_path)

    def write_tfrecords(data, data_name):
        full_path = processed_path + "/" + data_name + ".tfrecords"
        if not os.path.exists(full_path):

            writer = tf.python_io.TFRecordWriter(full_path)

            for fact in data:
                encoded_fact = encode(fact)
                # lengths[len(encoded_fact)] += 1
                fact_length = len(encoded_fact)
                padding = seq_len - fact_length

                if (padding >= 0):
                    feature = np.pad(encoded_fact, (0, padding), 'constant')
                    example = {}
                    example["input_ids"] = create_int_feature(feature)

                    tf_example = tf.train.Example(features=tf.train.Features(feature=example))
                    writer.write(tf_example.SerializeToString())

            writer.close()

    write_tfrecords(facts[:-1000], "training")
    write_tfrecords(facts[-1000:], "testing")
    write_tfrecords(facts[-100:], "predict")

    # Get the distribution on the length of each fact in tokens
    # for i, length in enumerate(lengths):
    #     print(str(i) + ": " + str(length))

    return vocab_size, tokenizer, longest_text


def get_tokenizer(texts):
    input_vocab_size = 2 ** 12

    # Create a BPE vocabulary using the abstracts

    return tfds.features.text.SubwordTextEncoder.build_from_corpus(
        texts, target_vocab_size=input_vocab_size)
