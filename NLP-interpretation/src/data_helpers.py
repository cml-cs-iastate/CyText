import re
import csv
import itertools
import pickle as pkl
import numpy as np
from collections import Counter


"""
Original taken from https://github.com/dennybritz/cnn-text-classification-tf
"""


def clean_str(string, remove_stopword):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = string.strip().lower()

    if remove_stopword:
        stop_words = {'ourselves', 'hers', 'between', 'yourself', 'there', 'about', 'having', 'with', 'they', 'an',
                      'be', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'itself', 'other', 'is', 's', 'am',
                      'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'are', 'we', 'these',
                      'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'himself', 'this', 'our', 'their',
                      'while', 'both', 'to', 'ours', 'had', 'she', 'all', 'when', 'at', 'any', 'them', 'and',
                      'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what',
                      'over', 'why', 'did', 'now', 'he', 'you', 'herself', 'has', 'where', 'myself', 'which', 'those',
                      'i', 'whom', 'being', 'if', 'theirs', 'my', 'a', 'by', 'doing', 'it', 'how', 'further', 'was',
                      'here', 'than', "'s"}
        string = ' '.join([w for w in string.split(' ') if w not in stop_words])

    return string


def load_data_and_labels(data_source, remove_stopword=False):
    """
    Loads data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Read the CSV file and get its contents
    with open(data_source, 'r', encoding='utf-8', errors='ignore') as f:
        csv_reader = csv.reader(f)
        # get the header
        header = next(csv_reader)
        label_idx = header.index('label')
        content_idx = header.index('content')
        print(f'The label index is : {label_idx} and the content index is : {content_idx}')

        y_text = list()
        x_text = list()

        for line in csv_reader:
            # get the sentence from the line
            sentence = line[content_idx].strip()
            x_text.append(sentence)
            y_text.append(int(line[label_idx]))

    # Split by words
    x_text = [clean_str(sent, remove_stopword) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]

    # get the lengths for every line
    lengths = np.array(list(map(len, [sent for sent in x_text])))

    return [x_text, y_text, lengths]


def pad_sentences(sentences, sequence_length=0, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    if sequence_length == 0:
        sequence_length = max(len(x) for x in sentences)

    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        if len(sentence) < sequence_length:
            num_padding = sequence_length - len(sentence)
            new_sentence = sentence + [padding_word] * num_padding
        else:
            new_sentence = sentence[:sequence_length]
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences, max_num_words):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences)).most_common()
    if max_num_words != 0 and max_num_words < len(word_counts):
        word_counts = word_counts[:max_num_words]

    # Mapping from index to word
    vocabulary_inv_list = [x[0] for x in word_counts]

    # Mapping from index to word
    vocabulary = {index: word for index, word in enumerate(vocabulary_inv_list)}

    return vocabulary


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    vocabulary_inv = {word: index for index, word in vocabulary.items()}
    x = np.array([[vocabulary_inv[word] if word in vocabulary_inv else 0 for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def load_data(data_source, sequence_length, vocabulary=None, max_num_words=0):
    """
    Loads and preprocessed data.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels, lengths = load_data_and_labels(data_source)
    sentences_padded = pad_sentences(sentences, sequence_length)
    if vocabulary is None:
        vocabulary = build_vocab(sentences_padded, max_num_words)
    else:
        with open(vocabulary, 'rb') as f:
            vocabulary = pkl.load(f, encoding='bytes')
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, lengths, vocabulary]


def train_test_split(data_source, test_split, sequence_length, vocabulary=None, max_num_words=0, shuffle=True):
    """
    :param data_source: path to the training dataset
    :param test_split: test ratio
    :param sequence_length: max length of the training documents
    :param vocabulary: None for training, load the one generated by training phase for testing
    :param max_num_words: the maximum number of words to be included in the vocabulary
    :param shuffle: True to shuffle the data, False otherwise
    """
    x, y, lengths, vocabulary = load_data(data_source, sequence_length, vocabulary, max_num_words)

    # Shuffle data
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x = x[shuffle_indices]
        y = y[shuffle_indices]
        lengths = lengths[shuffle_indices]

    train_len = int(len(x) * (1 - test_split))
    x_train = x[:train_len]
    y_train = y[:train_len]
    lengths_train = lengths[:train_len]
    x_test = x[train_len:]
    y_test = y[train_len:]
    lengths_test = lengths[train_len:]

    return x_train, y_train, lengths_train, x_test, y_test, lengths_test, vocabulary


def batch_iter(data, labels, lengths, batch_size, num_epochs):
    """
    A mini-batch iterator to generate mini-batches for training neural network
    :param data: a list of sentences. each sentence is a vector of integers
    :param labels: a list of labels
    :param lengths: lengths of the input samples
    :param batch_size: the size of mini-batch
    :param num_epochs: number of epochs
    :return: a mini-batch iterator
    """
    assert len(data) == len(labels) == len(lengths)

    data_size = len(data)
    epoch_length = int(data_size / batch_size) + 1
    print(f'Total number of batches per epoch: {epoch_length}')

    for _ in range(num_epochs):
        for batch_num in range(epoch_length):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)

            xdata = data[start_index: end_index]
            ydata = labels[start_index: end_index]
            ldata = lengths[start_index: end_index]

            yield xdata, ydata, ldata
