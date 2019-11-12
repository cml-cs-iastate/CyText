import re
import csv
import time
import numpy as np

from tensorflow.contrib import learn


def load_data(file_path, min_frequency=0, max_length=0, vocab_processor=None, shuffle=True, return_term_frequency=False):
    """
    Build dataset for mini-batch iterator
    :param file_path: Data file path
    :param min_frequency: the minimal frequency of words to keep
    :param max_length: the max document length
    :param vocab_processor: the predefined vocabulary processor
    :param shuffle: shuffle the dataset
    :param return_term_frequency: return words frequencies per class
    :return data, labels, lengths, vocabulary processor
    """
    start = time.time()
    term_frequency = dict()

    # Read the CSV file and get its contents
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        print('Building dataset ...')
        csv_reader = csv.reader(f)
        # get the header
        header = next(csv_reader)
        label_idx = header.index('label')
        content_idx = header.index('content')
        print(f'The label index is : {label_idx} and the content index is : {content_idx}')

        labels = list()
        contents = list()

        for line in csv_reader:
            # get the sentence from the line
            sentence = line[content_idx].strip()

            # convert to lower case letters
            sentence = sentence.lower()

            # remove special characters
            sentence = pre_processing(sentence)

            # remove the sentence if it has no words
            if len(sentence) < 1:
                continue

            # get the word frequency per class
            if return_term_frequency:
                sample_length = len(sentence) if len(sentence) < max_length else max_length
                for word in sentence.split()[:sample_length]:
                    term_frequency.setdefault(int(line[label_idx]), dict())
                    term_frequency[int(line[label_idx])][word] = term_frequency[int(line[label_idx])].setdefault(word, 0) + 1

            # append the sentence and label to the dataset
            contents.append(sentence)

            labels.append(int(line[label_idx]))

    labels = np.array(labels)
    # get the lengths for every line
    lengths = np.array(list(map(len, [sent.strip().split(' ') for sent in contents])))
    # print(f'The length of the sentences by order is : {lengths}')

    if max_length == 0:
        max_length = max(lengths)

    # Extract vocabulary from sentences and map words to indices
    if vocab_processor is None:
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_length, min_frequency=min_frequency)
        data = np.array(list(vocab_processor.fit_transform(contents)))
    else:
        data = np.array(list(vocab_processor.transform(contents)))

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(len(data)))
        data = data[shuffle_indices]
        labels = labels[shuffle_indices]
        lengths = lengths[shuffle_indices]

    end = time.time()

    assert len(data) == len(labels) == len(lengths)

    print('Dataset has been built successfully.')
    print('Run time: {}'.format(end - start))
    print(f'Number of samples: {len(data)}')
    print(f'Number of labels: {len(labels)}')
    print(f'Vocabulary size: {len(vocab_processor.vocabulary_._mapping)}')
    print(f'Max document length: {vocab_processor.max_document_length}\n')

    return data, labels, lengths, vocab_processor, term_frequency


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
    epoch_length = data_size // batch_size

    for _ in range(num_epochs):
        for i in range(epoch_length):
            start_index = i * batch_size
            end_index = start_index + batch_size

            xdata = data[start_index: end_index]
            ydata = labels[start_index: end_index]
            sequence_length = lengths[start_index: end_index]

            yield xdata, ydata, sequence_length


def pre_processing(line):
    # Remove special characters and stop words
    line = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", line)
    line = re.sub(r"\'s", " \'s", line)
    line = re.sub(r"\'ve", " \'ve", line)
    line = re.sub(r"n\'t", " n\'t", line)
    line = re.sub(r"\'re", " \'re", line)
    line = re.sub(r"\'d", " \'d", line)
    line = re.sub(r"\'ll", " \'ll", line)
    line = re.sub(r",", " , ", line)
    line = re.sub(r"!", " ! ", line)
    line = re.sub(r"\(", " \( ", line)
    line = re.sub(r"\)", " \) ", line)
    line = re.sub(r"\?", " \? ", line)
    line = re.sub(r"\s{2,}", " ", line)

    return line
