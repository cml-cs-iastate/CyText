import os
import sys
import csv
import random
import numpy as np
import matplotlib.pyplot as plt


def generate_ground_truth(input_x, input_y, words_vocab, if_dict):
    truth = []
    for idx, sample in enumerate(input_x):
        temp = []
        for word in sample:
            if word in words_vocab and input_y[idx] in if_dict:
                if words_vocab[word] in if_dict[input_y[idx]]:
                    temp.append(1)  # convert if_dict[input_y[idx]][word] for weight assignment
                else:
                    temp.append(0)
            else:
                temp.append(0)
        truth.append(temp)
    return truth


def plot_results(t_loss, t_accuracy, v_loss, v_accuracy,
                 j_t_loss, j_t_accuracy, j_v_loss, j_v_accuracy,
                 evaluate_every_steps, current_step, path):
    """
    """
    x = np.arange(evaluate_every_steps-1, current_step, evaluate_every_steps)
    plt.figure(1)
    plt.subplot(211)
    plt.plot(x, np.array(t_loss), 'r', label='Training Loss')
    plt.plot(x, np.array(v_loss), 'b', label='Validation Loss')
    plt.plot(x, np.array(j_t_loss), '--r', label='Jaccard Training Loss')
    plt.plot(x, np.array(j_v_loss), '--b', label='Jaccard Validation Loss')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(212)
    plt.plot(x, np.array(t_accuracy), 'r', label='Training Accuracy')
    plt.plot(x, np.array(v_accuracy), 'b', label='Validation Accuracy')
    plt.plot(x, np.array(j_t_accuracy), '--r', label='Jaccard Training Accuracy')
    plt.plot(x, np.array(j_v_accuracy), '--b', label='Jaccard Validation Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Steps')
    plt.legend()
    plt.draw()
    plt.savefig(os.path.join(path, 'figure_' + str(current_step) + '.png'), dpi=300)
    plt.close()


def random_remove_words(x_test_, real_lengths_, remove_ratio):
    """
    remove words randomly from input by random_percentage
    :param x_test_: the input data samples
    :param real_lengths_: the lengths of the input data samples
    :param remove_ratio: the ratio of the words to be remove from the real length of the input sample [0, 1]
    :return:
    """
    for doc_idx in range(len(x_test_)):
        remove_list = random.sample(range(real_lengths_[doc_idx]), int(real_lengths_[doc_idx] * remove_ratio))
        for index in remove_list:
            x_test_[doc_idx][index] = 0
    return x_test_


def uniform_remove_word(x_test_, real_lengths_, remove_ratio, heat_map_, most_relevant_):
    """
    remove top/bottom words from input based on their important values from the heat-map
    :param x_test_: the input data samples
    :param real_lengths_: the lengths of the input data samples
    :param remove_ratio: the ratio of the words to be remove from the real length of the input sample [0, 1]
    :param heat_map_: contains the importance value for each word
    :param most_relevant_: if True, remove the most relevant words, otherwise, remove the least relevant words
    :return:
    """
    for doc_idx in range(len(x_test_)):
        for i in range(int(real_lengths_[doc_idx] * remove_ratio)):
            # find the most or least relevant word
            if most_relevant_:
                word_idx = np.argmax(heat_map_[doc_idx])
                heat_map_[doc_idx][word_idx] = -sys.maxsize - 1
            else:
                word_idx = np.argmin(heat_map_[doc_idx])
                heat_map_[doc_idx][word_idx] = sys.maxsize
            # remove the word
            x_test_[doc_idx][word_idx] = 0
    return x_test_


def average_drop(logits, logits_after_removal, predictions, batch_size):
    avg_drop = 0
    for batch_idx in range(len(logits)):
        for doc_idx in range(batch_size):
            avg_drop += max(0, logits[batch_idx][doc_idx][predictions[batch_idx][doc_idx]] - logits_after_removal[batch_idx][doc_idx][predictions[batch_idx][doc_idx]]) / logits[batch_idx][doc_idx][predictions[batch_idx][doc_idx]]
    return avg_drop / (len(logits)*batch_size)


def increase_confidence(logits, logits_after_removal, predictions, batch_size):
    increase_conf = 0
    for batch_idx in range(len(logits)):
        for doc_idx in range(batch_size):
            increase_conf += 1 if logits[batch_idx][doc_idx][predictions[batch_idx][doc_idx]] < logits_after_removal[batch_idx][doc_idx][predictions[batch_idx][doc_idx]] else 0
    return increase_conf / (len(logits)*batch_size)


def precision_and_recall(prediction_file):
    # Read the CSV file and get its contents
    with open(prediction_file, 'r', encoding='utf-8', errors='ignore') as f:
        csv_reader = csv.reader(f)
        header = next(csv_reader)
        true_class = header.index('True class')
        prediction = header.index('Prediction')

        tp, fn, fp = 0, 0, 0
        for line in csv_reader:
            if line[true_class] == '1' and line[prediction] == '1.0':
                tp += 1
            elif line[true_class] == '1' and line[prediction] == '0.0':
                fn += 1
            elif line[true_class] == '0' and line[prediction] == '1.0':
                fp += 1

        precision = tp / ((tp + fp) + 1e-10)
        recall = tp / ((tp + fn) + 1e-10)

        return precision, recall
