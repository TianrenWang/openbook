import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from sklearn.utils import shuffle
import os
import pandas as pd
import csv


def create_int_feature(values):
    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return f


def text_processor(data_path, seq_len, vocab_level, processed_path):
    samples = []
    facts_indicator = []
    dir_path = data_path

    def accumulate_samples(sample_list, fact_list, filename, facts=False):

        data = open(dir_path + filename, "r")
        line = data.readline().capitalize()

        while line:
            sample_list.append(str.encode(line[:-2]))
            fact_list.append(facts)
            line = data.readline()

        data.close()

    accumulate_samples(samples, facts_indicator, "openbook_facts.txt", True)
    accumulate_samples(samples, facts_indicator, "AristoTable.txt")
    accumulate_samples(samples, facts_indicator, "Annotation.txt")
    accumulate_samples(samples, facts_indicator, "scitail.txt")
    accumulate_samples(samples, facts_indicator, "quartz.txt")
    accumulate_samples(samples, facts_indicator, "sciq.txt")

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

    train_facts = samples[:-2000]
    train_facts_indictator = facts_indicator[:-2000]

    train_facts = [train_facts[i] for i in range(len(train_facts_indictator)) if train_facts_indictator[i]]
    train_facts_indictator = [i for i in train_facts_indictator if i]

    test_facts = samples[-2000:]
    test_facts_indictator = facts_indicator[-2000:]

    test_facts = [test_facts[i] for i in range(len(test_facts_indictator)) if test_facts_indictator[i]]
    test_facts_indictator = [i for i in test_facts_indictator if i]

    write_tfrecords(train_facts, train_facts_indictator, "facts_only_training")
    write_tfrecords(test_facts, test_facts_indictator, "facts_only_testing")

    # Get the distribution on the length of each fact in tokens
    # for i, length in enumerate(lengths):
    #     print(str(i) + ": " + str(length))

    # This dataset will be used to test how well the connections form between concepts
    samples = []
    facts_indicator = []
    accumulate_samples(samples, facts_indicator, "connection.txt", True)
    write_tfrecords(samples, facts_indicator, "connections")

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

def openbook_question_processor(word_embedding_path, processed_path, max_length):

    # Get the list of words in GloVE
    full_word_dict = {}
    full_word_embedding = []
    vector_sum = 0
    decoder = []
    actual_embedding = []
    actual_dict = {}

    data = open(word_embedding_path, "r")
    line = data.readline()

    print("Getting Embedded Words")
    counter = 0

    while line:
        if counter % 100000 == 0:
            print(counter)

        values = line.strip().split()
        print(values)
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        vector_sum += vector
        full_word_dict[word] = (counter, vector)
        # decoder.append(word)
        # full_word_embedding.append(vector)
        line = data.readline()

        counter += 1

    data.close()

    actual_embedding.append(vector_sum/counter) # this is for the unknown word
    actual_embedding.append(np.zeros([300], dtype='float32')) # this is for padding tokens
    decoder.append("<unk>")  # this is for the unknown word
    decoder.append("<pad>")  # this is for padding tokens

    def _add_word(word, vector, dict_len):
        if word in actual_dict:
            return actual_dict[word]
        else:
            decoder.append(word)
            actual_dict[word] = dict_len[0]
            actual_embedding.append(vector)
            dict_len[0] += 1
            return dict_len[0] - 1

    if not os.path.exists(processed_path):
        os.makedirs(processed_path)

    def write_tfrecords(data_path, data_name):
        max = 0
        full_path = processed_path + "/" + data_name + ".tfrecords"
        embedding_size = [len(actual_embedding)]

        if not os.path.exists(full_path):

            data = pd.read_json(data_path, lines=True)
            answers = data['answerKey']
            questions = data['question']

            writer = tf.io.TFRecordWriter(full_path)

            for question, answer in zip(questions, answers):
                choices = question['choices']
                # print("choices: " + str(choices))
                choices_tokens = []
                stem = []

                question_stem = question['stem'].lower().split()
                # print("stem: " + str(question_stem))

                for word in question_stem:
                    if word in full_word_dict:
                        stem.append(_add_word(word, full_word_dict[word][1], embedding_size))
                    else:
                        stem.append(0)

                for i in range(len(choices)):
                    choice = []
                    words = choices[i]['text'].lower().split()
                    for word in words:
                        # print(word)
                        if word[0] == '\'' and word[-1] == '\'':
                            word = word[1:-1]
                        if word in full_word_dict:
                            word_index = _add_word(word, full_word_dict[word][1], embedding_size)
                            choice.append(word_index)
                        elif '\'' in word:
                            index = word.index('\'')
                            word1 = word[:index]
                            word2 = word[index:]
                            if word1 in full_word_dict:
                                word_index1 = _add_word(word1, full_word_dict[word1][1], embedding_size)
                                choice.append(word_index1)
                            else:
                                choice.append(0)
                            if word2 in full_word_dict:
                                word_index2 = _add_word(word2, full_word_dict[word2][1], embedding_size)
                                choice.append(word_index2)
                            else:
                                choice.append(0)
                        else:
                            choice.append(0)
                    full_choice = stem + choice

                    # print("full choice: " + str(full_choice))
                    # print("Decoded: " + str([decoder[i] for i in list(full_choice)]))

                    choice_length = len(full_choice)
                    if max < choice_length:
                        max = choice_length
                    padding = max_length - choice_length
                    full_choice = np.pad(full_choice, (0, padding), 'constant', constant_values=(0, 1))
                    # print("full_choice: " + str(full_choice))
                    # print("choices_tokens: " + str(choices_tokens))
                    choices_tokens += list(full_choice)
                answerValue = 0
                if answer == 'B':
                    answerValue = 1
                elif answer == 'C':
                    answerValue = 2
                elif answer == 'D':
                    answerValue = 3

                example = {}
                example["input_ids"] = create_int_feature(choices_tokens)
                example["answer_id"] = create_int_feature([answerValue])

                tf_example = tf.train.Example(features=tf.train.Features(feature=example))
                writer.write(tf_example.SerializeToString())

            writer.close()
            print("Maximum choice length: " + str(max))

    write_tfrecords("data/train.jsonl", "training_questions")
    write_tfrecords("data/test.jsonl", "testing_questions")
    write_tfrecords("data/dev.jsonl", "validating_questions")

    print("Decoder: " + str(len(decoder)))

    return np.array(actual_embedding), decoder

def relationship_processor(embedding_path, data_path):

    # Process the word embedding
    # Get the list of words in GloVE
    vectors = {}
    indices = {}

    data = open(embedding_path, "r")
    line = data.readline()

    print("Getting Embedded Words")
    counter = 1

    while line:
        values = line.strip().split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        vectors[word] = vector
        indices[word] = counter
        line = data.readline()

        counter += 1

    data.close()

    print("Completed Processing Embedded Words")

    def get_relationships(filename):

        concepts = []
        connections = []
        originals = []

        notignored = 0

        with open(filename, newline='') as f:
            reader = csv.reader(f)
            for relationship in reader:
                if len(relationship) > 1 or len(relationship) % 2 == 0:

                    try: # There are spelling mistakes that lead to keyerror that I am just going to ignore
                        current_concepts = []
                        current_connections = [[]]

                        # Process each concepts in the relationship
                        for i in range(len(relationship)):
                            if i % 2 == 0:
                                concept = relationship[i].lower().split()
                                concept_vector = 0
                                for component in concept:
                                    concept_vector += vectors[component]
                                concept_vector /= len(concept)
                                current_concepts.append(concept_vector)
                            else:
                                if relationship[i] == '':
                                    current_connections.append(current_connections[-1])
                                else:
                                    connection = relationship[i].lower().split()
                                    connection_vector = 0
                                    for component in connection:
                                        connection_vector += vectors[component]
                                    connection_vector /= len(connection)
                                    current_connections.append(connection_vector)

                        notignored += 1
                        connections = connections + current_connections
                        concepts = concepts + current_concepts
                        originals.append(relationship)
                    except:
                        pass

        print("processed samples: " + str(notignored))
        return concepts, connections, originals

    concepts, connections, originals = get_relationships(data_path)

    # concepts: an array where each element is a concept from a relationship
    # connections: an array where each element is a relationship connecting the i - 1th and ith concepts
    # originals: an array where each element is the original relationship in string

    print("concepts: " + str(np.shape(concepts)))
    print("connections: " + str(np.shape(connections)))
    print("originals: " + str(np.shape(originals)))

    return concepts, connections, originals
