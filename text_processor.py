import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from sklearn.utils import shuffle
import os


def create_int_feature(values):
    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return f

def text_processor(data_path, training_file):
    filepath = data_path + "/" + os.listdir(data_path)[0]

    data = open(filepath, "r")
    line = data.readline()
    facts = []

    while line:
        facts.append(str.encode(line))
        line = data.readline()

    data.close()

    tokenizer = get_tokenizer(facts)

    # example = tokenizer.encode("An example of camouflage is when something changes color in order to have the same color as its environment")

    # for token in example:
    #     print(tokenizer.decode([token]))

    vocab_size = tokenizer.vocab_size + 2

    def encode(facts):
        """Turns an abstract in English into BPE (Byte Pair Encoding).
        Adds start and end token to the abstract.

        Keyword arguments:
        abstract -- the abstract (type: bytes)
        """

        encoded_facts = [tokenizer.vocab_size] + tokenizer.encode(facts) + [tokenizer.vocab_size + 1]

        return encoded_facts

    MAX_LENGTH = 40

    # Create a list of encoded abstracts
    encoded_facts = []

    # lengths = [0] * 200

    if not os.path.exists(training_file):

        writer = tf.python_io.TFRecordWriter(training_file)

        for fact in facts:
            encoded_fact = encode(fact)
            # lengths[len(encoded_fact)] += 1
            fact_length = len(encoded_fact)
            padding = MAX_LENGTH - fact_length

            if (padding >= 0):
                feature = np.pad(encoded_fact, (0, padding), 'constant')
                example = {}
                example["input_ids"] = create_int_feature(feature)
                encoded_facts.append(feature)

                tf_example = tf.train.Example(features=tf.train.Features(feature=example))
                writer.write(tf_example.SerializeToString())

        writer.close()

    # Get the distribution on the length of each fact in tokens
    # for i, length in enumerate(lengths):
    #     print(str(i) + ": " + str(length))

    return shuffle(np.array(encoded_facts)), vocab_size, tokenizer


def get_tokenizer(texts):
    input_vocab_size = 2 ** 11

    # Create a BPE vocabulary using the abstracts

    return tfds.features.text.SubwordTextEncoder.build_from_corpus(
        texts, target_vocab_size=input_vocab_size)