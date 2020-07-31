import itertools
import pickle as pkl

from collections import Counter
from mip import Model, xsum, minimize, BINARY, OptimizationStatus

import src.settings as settings
from src.data_helpers import load_data_and_labels


# Tested
def milp(data, vocabulary, num_classes, vocab_size):
    """
    Mixed-Integer Linear Programing
    :param data: matrix of one-hot representation of each document per class       shape {num_classes: [None, vocab_size]}
    :param vocabulary: the super vector of the unique words
    :param num_classes: the number of classes
    :param vocab_size: the vocabulary size in terms number of words
    """

    # Create the optimization model
    model = Model()
    print('Optimization model is done!')

    # IMLP Variable: (to optimize)  shape [num_classes, vocab_size]
    V = [[model.add_var(var_type=BINARY) for _ in range(vocab_size)] for _ in range(num_classes)]
    print('Optimization variables are done!')

    # Optimization function
    # xsum(V[c][j] for c in range(num_classes) for j in range(vocab_size)) +
    model.objective = minimize(xsum(d[j] * V[c_prime][j]
                                    for c in range(num_classes)
                                    for d in data[c]
                                    for c_prime in range(num_classes)
                                    if c != c_prime
                                    for j in range(vocab_size)))
    print('Optimization objective function is done!')

    # Constraints
    # Equation (5)
    """
    Each interpretation feature must belong to only one category (Tested)
    """
    for j in range(vocab_size):
        model.add_constr(xsum(V[c][j] for c in range(num_classes)) <= 1)
    print('Optimization subjective equation (5) is done!')

    # Equation (6)
    """
    Each document of class c must have at least one interpretation feature that belong to class c (Tested)
    """
    for c in range(num_classes):
        for d in data[c]:
            model.add_constr(xsum(d[j] * V[c][j] for j in range(vocab_size)) >= 1)
    print('Optimization subjective equation (6) is done!')

    # run optimization
    status = model.optimize()
    print('Optimization is over!')

    # print results
    if status == OptimizationStatus.OPTIMAL:
        print('optimal solution cost {} found'.format(model.objective_value))
    elif status == OptimizationStatus.FEASIBLE:
        print('sol.cost {} found, best possible: {}'.format(model.objective_value, model))
    elif status == OptimizationStatus.NO_SOLUTION_FOUND:
        print('no feasible solution found, lower bound is: {}'.format(model.objective_bound))

    milp_if = dict()
    if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
        for i in range(num_classes):
            milp_if.setdefault(i, set())
            for j in range(vocab_size):
                if V[i][j].x > 0:
                    milp_if[i].add(vocabulary[j])
    return milp_if


# Tested
def one_hot_doc(data_source, max_num_words):
    """
    generate one-hot representation for each document per class
    :param data_source: path to the training dataset
    :param max_num_words: maximum number of word in vocabulary dict
    """
    # Load and preprocess data
    sentences, labels, lengths = load_data_and_labels(data_source, remove_stopword=True)
    print('Load data is done!')

    word_counts = Counter(itertools.chain(*sentences)).most_common()
    if max_num_words != 0 and max_num_words < len(word_counts):
        word_counts = word_counts[:max_num_words]
    vocab = [w[0] for w in word_counts]
    print('Building vocab is done!')

    sentences_vectors = dict()
    for idx, sentence in enumerate(sentences):
        sent_vec = []
        for token in vocab:
            sent_vec.append(1 if token in sentence else 0)
        sentences_vectors.setdefault(labels[idx], list())
        if sum(sent_vec) > 0:
            sentences_vectors[labels[idx]].append(sent_vec)
    print('Building one-hot representation is done!')

    return sentences_vectors, vocab


# ======================================================================================================================
# Running the code
data_, vocabulary_ = one_hot_doc(settings.data_source, max_num_words=settings.max_words)
del data_[0]
del data_[3]

milp_if_ = milp(data=data_, vocabulary=vocabulary_, num_classes=settings.num_classes, vocab_size=settings.max_words)

with open(settings.milp_vocab, 'wb') as f:
    pkl.dump(vocabulary_, f, pkl.HIGHEST_PROTOCOL)

with open(settings.milp_IF, 'wb') as f:
    pkl.dump(milp_if_, f, pkl.HIGHEST_PROTOCOL)
