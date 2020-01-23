import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from sklearn.utils import shuffle
import os
import pandas as pd


def create_int_feature(values):
    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return f


def text_processor(data_path, seq_len, vocab_level, processed_path):
    samples = []
    facts_indicator = []
    dir_path = data_path

    def accumulate_samples(filename, facts=False):

        data = open(dir_path + filename, "r")
        line = data.readline().capitalize()

        while line:
            samples.append(str.encode(line[:-2]))
            facts_indicator.append(facts)
            line = data.readline()

        data.close()

    accumulate_samples("openbook_facts.txt", True)
    accumulate_samples("AristoTable.txt")
    accumulate_samples("Annotation.txt")
    accumulate_samples("scitail.txt")
    accumulate_samples("quartz.txt")
    accumulate_samples("sciq.txt")

    facts_indicator, samples = shuffle(facts_indicator, samples)

    tokenizer = get_tokenizer(samples, vocab_level)

    vocab_size = tokenizer.vocab_size + 2

    def encode(sample):
        """Turns an abstract in English into BPE (Byte Pair Encoding).
        Adds start and end token to the abstract.

        Keyword arguments:
        abstract -- the abstract (type: bytes)
        """

        encoded_sample = [tokenizer.vocab_size] + tokenizer.encode(sample) + [tokenizer.vocab_size + 1]

        return encoded_sample

    # lengths = [0] * 200

    if not os.path.exists(processed_path):
        os.makedirs(processed_path)

    def write_tfrecords(samples, facts, data_name):
        full_path = processed_path + "/" + data_name + ".tfrecords"
        if not os.path.exists(full_path):

            writer = tf.io.TFRecordWriter(full_path)

            for sample, fact in zip(samples, facts):
                encoded_fact = encode(sample)
                # lengths[len(encoded_fact)] += 1
                fact_length = len(encoded_fact)
                padding = seq_len - fact_length

                if (padding >= 0 and fact_length >= 3):
                    feature = np.pad(encoded_fact, (0, padding), 'constant')
                    example = {}
                    example["input_ids"] = create_int_feature(feature)
                    example["input_len"] = create_int_feature([fact_length])
                    example["input_fact"] = create_int_feature([fact])

                    tf_example = tf.train.Example(features=tf.train.Features(feature=example))
                    writer.write(tf_example.SerializeToString())

            writer.close()

    write_tfrecords(samples[:-2000], facts_indicator[:-2000], "training")
    write_tfrecords(samples[-2000:], facts_indicator[-2000:], "testing")
    write_tfrecords(samples[-2000:], facts_indicator[-2000:], "predict")

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
