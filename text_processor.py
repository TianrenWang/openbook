import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from sklearn.utils import shuffle
from random import shuffle
import os
import pandas as pd


def create_int_feature(values):
    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return f


def text_processor(data_path, seq_len, vocab_level, processed_path):
    facts = []
    dir_path = data_path

    def accumulate_facts(filename):

        data = open(dir_path + filename, "r")
        line = data.readline().capitalize()

        while line:
            facts.append(str.encode(line[:-1]))
            line = data.readline()

        data.close()

    accumulate_facts("openbook_facts.txt")
    accumulate_facts("AristoTable.txt")
    accumulate_facts("Annotation.txt")
    accumulate_facts("scitail.txt")
    accumulate_facts("quartz.txt")
    accumulate_facts("sciq.txt")

    shuffle(facts)

    tokenizer = get_tokenizer(facts, vocab_level)

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

            writer = tf.io.TFRecordWriter(full_path)

            for fact in data:
                encoded_fact = encode(fact)
                # lengths[len(encoded_fact)] += 1
                fact_length = len(encoded_fact)
                padding = seq_len - fact_length

                if (padding >= 0 and fact_length >= 3):
                    feature = np.pad(encoded_fact, (0, padding), 'constant')
                    example = {}
                    example["input_ids"] = create_int_feature(feature)
                    example["input_len"] = create_int_feature([fact_length])

                    tf_example = tf.train.Example(features=tf.train.Features(feature=example))
                    writer.write(tf_example.SerializeToString())

            writer.close()

    write_tfrecords(facts[:-1000], "training")
    write_tfrecords(facts[-1000:], "testing")
    write_tfrecords(facts[-1000:], "predict")

    # Get the distribution on the length of each fact in tokens
    # for i, length in enumerate(lengths):
    #     print(str(i) + ": " + str(length))

    return vocab_size, tokenizer


def get_tokenizer(texts, vocab_level):
    input_vocab_size = 2 ** vocab_level

    # Create a BPE vocabulary using the abstracts

    return tfds.features.text.SubwordTextEncoder.build_from_corpus(
        texts, target_vocab_size=input_vocab_size)


def openbook_processor():
    data = pd.read_json("data/train.jsonl", lines=True)
    answers = data['answerKey']
    questions = data['question']
    f = open("data/openbook_train.txt", "a")

    for question, answer in zip(questions, answers):
        choices = question['choices']

        for choice in choices:
            if answer == choice['label']:
                correctChoice = choice['text']

        fact = question['stem'] + " " + correctChoice + "\n"
        f.write(fact)
    f.close()
