import csv
import numpy as np
import pickle as pkl
import tensorflow as tf

from tensorflow.keras import models

import src.settings as settings
from src.utils import generate_ground_truth, uniform_remove_word, random_remove_words
from src.interpretation import get_interpretation_de
from src.data_helpers import train_test_split, batch_iter

tf.compat.v1.disable_eager_execution()


# Tested
def table_1(if_dict):
    """
    if_dict
    return the number of pseudo ground truth interpretation features per class
    :param if_dict: the dictionary of the interpretation features
    """
    with open(if_dict, 'rb') as f:
        if_dict = pkl.load(f)

    for label, if_set in if_dict.items():
        print(f'The number of interpretation features of class {label} is {len(if_set)}')


# Tested
def compare(expert, if_dict, data, labels):
    """
    :param expert: list of the interpretation features by expert        shape [250, None]
    :param if_dict: the dictionary of the interpretation features
    :param data: the list of the 250 Amazon review                      shape [250, None]
    :param labels: the 250 label of the data samples                    shape [250, None]
    """
    # compare between entity_1 and entity_2 (Human with Human / or Human with Machine)
    a, b, c, d, counter = 0, 0, 0, 0, 0
    for idx, line in enumerate(data):
        label = labels[idx]

        # TODO: For MR Dataset only
        if label == 0:
            l1 = 1
            l2 = 2
        else:
            l1 = 4
            l2 = 5

        for word in line.strip().split():
            counter += 1
            if word in expert[idx] and (word in if_dict[l1] or word in if_dict[l2]):
                a += 1
            elif word in expert[idx] and (word not in if_dict[l1] and word not in if_dict[l2]):
                b += 1
            elif word not in expert[idx] and (word in if_dict[l1] or word in if_dict[l2]):
                c += 1
            elif word not in expert[idx] and (word not in if_dict[l1] and word not in if_dict[l2]):
                d += 1

        po = (a + d) / (a + b + c + d)
        p1 = ((a + b) / (a + b + c + d)) * ((a + c) / (a + b + c + d))
        p0 = ((d + b) / (a + b + c + d)) * ((d + c) / (a + b + c + d))
        pe = p1 + p0
        k = (po - pe) / (1 - pe)

    print(f'a: {a}, b: {b}, c: {c}, d: {d}, sum(a,b,c,d): {a + b + c + d}, len(data): {counter}')
    print(f'po: {po}, p0: {p0}, p1: {p1}, pe: {pe}, Kappa: {k}')


# Tested
def table_2(if_dict):
    """
    250 Amazon user-review
    Calculate the the Cohen’s kappa inter-agreement rating of 250 user reviews of Amazon dataset.
    :param if_dict: the dictionary of the interpretation features
    """

    with open('data/Interagreement/Amazon_250_review.csv', 'r', encoding='utf-8', errors='ignore') as f:
        csv_reader = csv.reader(f)
        header = next(csv_reader)
        label_idx = header.index('label')
        content_idx = header.index('content')
        data = list()
        labels = list()
        for sample in csv_reader:
            data.append(sample[content_idx])
            labels.append(int(sample[label_idx]))

    # load Natalie's data
    with open('data/Interagreement/Natalie.csv', 'r', encoding='utf-8', errors='ignore') as f:
        csv_reader = csv.reader(f)
        natalie = list()
        for sample in csv_reader:
            natalie.append(sample[0].strip().split())

    # load Carrie's data
    with open('data/Interagreement/Carrie.csv', 'r', encoding='utf-8', errors='ignore') as f:
        csv_reader = csv.reader(f)
        carrie = list()
        for sample in csv_reader:
            carrie.append(sample[0].strip().split())

    # load interpretation features dictionary
    with open(if_dict, 'rb') as f:
        if_dict = pkl.load(f)

    compare(natalie, if_dict, data, labels)
    compare(carrie, if_dict, data, labels)


# Tested
def sort_second(val):
    """
    function to return the second element of the two elements passed as the parameter
    """
    return val[1]


# Tested
def figure_2(if_dict, n):
    """
    if_dict
    Return the ten most relevant interpretation features from Hybrid-PGT. The interpretation features are listed in
    decreasing of their relevance score to each class.
    :param if_dict: the dictionary of the interpretation features
    :param n: top n interpretation features
    """
    with open(if_dict, 'rb') as f:
        if_dict = pkl.load(f)

    for label, if_set in if_dict.items():
        temp = list()
        for if_word, relevance_value in if_set.items():
            temp.append((if_word, relevance_value))

        # sorts the array in descending according to second element
        temp.sort(key=sort_second, reverse=True)
        print(f'The most relevance {n} interpretation features of class {label} are : {temp[:n] if n < len(temp) else temp}')


# Tested
def table_3(if_dict, vocab_dict_path, input_model_path, num_classes, max_length, threshold=0.3, interpretation_method='grad*input'):
    """
    MR Dataset
    Return the interpretation effectiveness using different PGTs on the MR dataset in terms of Kappa enter-agreement,
    interpretation precision and recall
    :param if_dict: the dictionary of the interpretation features
    :param vocab_dict_path: the vocabulary dictionary of the trained model
    :param input_model_path: the standard cnn model trained normally on the MR dataset
    :param num_classes: settings.num_classes of the dataset
    :param max_length: maximum sequence length of the documents from settings.sequence_length
    :param threshold: the threshold for the interpretation feature to be considered
    :param interpretation_method:
    """
    precision = 0
    recall = 0
    kappa = 0

    with open(if_dict, 'rb') as f:
        if_dict = pkl.load(f)

    x_train, y_train, l_train, _, _, _, vocab_dict = train_test_split(settings.data_source,
                                                                      settings.test_split,
                                                                      settings.sequence_length,
                                                                      vocabulary=vocab_dict_path)

    # Get the heatmap interpretation of the interpretation method
    interpretation_heatmap, y_prediction = get_interpretation_de(input_model_path, x_train, y_train, num_classes, interpretation_method=interpretation_method)

    #  real sample length
    real_length = [l if l < max_length else max_length for l in l_train]

    counter = 0
    for sample_idx in range(len(x_train)):
        if y_train[sample_idx] == y_prediction[sample_idx]:
            counter += 1
            a, b, c, d = 0, 0, 0, 0

            for word_idx in range(real_length[sample_idx]):

                # TODO: For MR Dataset only
                if y_train[sample_idx] == 0:
                    l1 = 1
                    l2 = 2
                else:
                    l1 = 4
                    l2 = 5

                word_relevance_score = 0
                if x_train[sample_idx][word_idx] in vocab_dict and (l1 in if_dict or l2 in if_dict):  # y_train[sample_idx] in if_dict:
                    if l1 in if_dict:  # TODO: no need for this 'if' or 'else' if we are using 5 classes
                        if vocab_dict[x_train[sample_idx][word_idx]] in if_dict[l1]:
                            word_relevance_score = 1
                    elif l2 in if_dict:
                        if vocab_dict[x_train[sample_idx][word_idx]] in if_dict[l2]:
                            word_relevance_score = 1

                if word_relevance_score == 1 and interpretation_heatmap[sample_idx][word_idx] >= threshold:
                    a += 1  # TP
                elif word_relevance_score == 1 and interpretation_heatmap[sample_idx][word_idx] < threshold:
                    b += 1  # FN
                elif word_relevance_score == 0 and interpretation_heatmap[sample_idx][word_idx] >= threshold:
                    c += 1  # FP
                elif word_relevance_score == 0 and interpretation_heatmap[sample_idx][word_idx] < threshold:
                    d += 1  # TN

            # Precision and Recall
            precision += a / ((a + c) + 1e-10)
            recall += a / ((a + b) + 1e-10)
            # Kappa
            po = (a + d) / (a + b + c + d)
            p1 = ((a + b) / (a + b + c + d)) * ((a + c) / (a + b + c + d))
            p0 = ((d + b) / (a + b + c + d)) * ((d + c) / (a + b + c + d))
            pe = p1 + p0
            kappa += (po - pe) / ((1 - pe) + 1e-10)

    precision /= counter
    recall /= counter
    kappa /= counter

    print(f'Interpretation precision: {precision}, Interpretation recall: {recall}, Kappa: {kappa}')


# Tested
def perturb_input(x_data, y_data, l_data, if_dict, vocab_dict, max_length, random, most_relevant, remove_ratio):
    """
    """
    # real sample length
    real_length = [l if l < max_length else max_length for l in l_data]

    # mark the interpretation ground truth for the document words
    interpretation_ground_truth = generate_ground_truth(x_data, y_data, vocab_dict, if_dict, binary=False)

    # remove words from input
    if random:
        x_perturb = random_remove_words(x_data, real_length, remove_ratio)
    else:
        x_perturb = uniform_remove_word(x_data,
                                        real_length,
                                        remove_ratio,
                                        heat_map=interpretation_ground_truth,
                                        most_relevant=most_relevant)

    return x_perturb


# Tested
def figure_4_intrinsic_validation(if_dict, vocab_dict_path, input_model_path, max_length, random=False, remove_ratio=0.5):
    """
    TODO: this one is designed only for the interpretation feature ground truth. We should also design one for each interpretation method.
    """
    with open(if_dict, 'rb') as f:
        if_dict = pkl.load(f)

    input_model = models.load_model(input_model_path)

    x_train, y_train, l_train, _, _, _, vocab_dict = train_test_split(settings.data_source,
                                                                      settings.test_split,
                                                                      settings.sequence_length,
                                                                      vocabulary=vocab_dict_path)

    batches = batch_iter(x_train, y_train, l_train, settings.batch_size, num_epochs=1)

    avg_drop = 0

    for batch in batches:
        x_data, y_data, l_data = batch

        predictions = input_model.predict(x_data)

        # Only change random and remove_ratio
        """keep most_relevant unchanged"""
        x_perturb = perturb_input(x_data, y_data, l_data, if_dict, vocab_dict, max_length, random=random, most_relevant=True, remove_ratio=remove_ratio)

        predictions_perturb = input_model.predict(x_perturb)

        for doc_idx in range(len(x_data)):
            label = np.argmax(predictions[doc_idx])
            avg_drop += max(0, predictions[doc_idx][label] - predictions_perturb[doc_idx][label]) / predictions[doc_idx][label]

    return avg_drop / len(x_train)


# Tested
def figure_4_increase_confidence(if_dict, vocab_dict, input_model_path, max_length, random=False, remove_ratio=0.5):
    """
    TODO: this one is designed only for the interpretation feature ground truth. We should also design one for each interpretation method.
    """
    with open(if_dict, 'rb') as f:
        if_dict = pkl.load(f)

    input_model = models.load_model(input_model_path)

    x_train, y_train, l_train, _, _, _, _ = train_test_split(settings.data_source,
                                                             settings.test_split,
                                                             settings.sequence_length,
                                                             vocabulary=vocab_dict)

    batches = batch_iter(x_train, y_train, l_train, settings.batch_size, num_epochs=1)

    increase_conf = 0

    for batch in batches:
        x_data, y_data, l_data = batch

        predictions = input_model.predict(x_data)

        # Only change random and remove_ratio
        """keep most_relevant unchanged"""
        x_perturb = perturb_input(x_data, y_data, l_data, if_dict, vocab_dict, max_length, random=random, most_relevant=False, remove_ratio=remove_ratio)

        predictions_perturb = input_model.predict(x_perturb)

        for doc_idx in range(len(x_data)):
            label = np.argmax(predictions[doc_idx])
            increase_conf += 1 if predictions[doc_idx][label] < predictions_perturb[doc_idx][label] else 0

        return increase_conf / len(x_train)


# ======================================================================================================================
# Running the code

# table_1(settings.post_processed_IF)

# table_2(settings.post_processed_IF)

# figure_2(settings.post_processed_IF, 10)

# {'saliency', 'grad*input', 'intgrad', 'elrp', 'deeplift'}
table_3(settings.post_processed_IF,
        settings.vocabulary_dict,
        settings.model_path,
        settings.num_classes,
        settings.sequence_length,
        threshold=0.2,
        interpretation_method='elrp')

# a = figure_4_intrinsic_validation(settings.post_processed_IF,
#                                   settings.vocabulary_dict,
#                                   settings.model_path,
#                                   settings.sequence_length,
#                                   random=False,
#                                   remove_ratio=0.9)
#
# b = figure_4_increase_confidence(settings.post_processed_IF,
#                                  settings.vocabulary_dict,
#                                  settings.model_path,
#                                  settings.sequence_length,
#                                  random=False,
#                                  remove_ratio=0.9)
#
# print(a, b)
