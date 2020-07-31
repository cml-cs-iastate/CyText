import numpy as np
import pickle as pkl
import tensorflow as tf

from operator import itemgetter
from tensorflow.keras import models, backend as K

import src.settings as settings
from src.data_helpers import train_test_split, batch_iter
from src.interpretation import get_interpretation

# Import DeepExplain
from deepexplain.tensorflow import DeepExplain


# Tested
def generate_if_dict(vocab_dict, input_model_path, window_size, target_layer_name, max_length):
    """
    This function generate interpretation features using the dataset that has been trained on the same dataset that we
    need to generate the interpretation features from.
    IMPORTANT: This function need to run on the Amazon_Yelp dataset on the server.
    """
    tf.compat.v1.disable_eager_execution()

    if_dict = dict()  # the summation of the relevance scores of each interpretation feature in each document
    if_count = dict()  # the count of each interpretation feature
    if_average = dict()  # average relevance score for each interpretation features

    input_model = models.load_model(input_model_path)

    x_train, y_train, l_train, _, _, _, vocab_dict = train_test_split(settings.data_source,
                                                                      settings.test_split,
                                                                      settings.sequence_length,
                                                                      vocabulary=vocab_dict)
    print('Loading data is over!')

    batches = batch_iter(x_train, y_train, l_train, settings.batch_size, num_epochs=1)

    y_prediction = input_model.predict(x_train)
    y_prediction = np.argmax(y_prediction, axis=1)
    print('Classification is over!')

    for batch_idx, batch in enumerate(batches):
        x_data, y_data, l_data = batch

        # Get the heatmap interpretation of the interpretation method
        interpretation_heatmap = get_interpretation('grad_cam', input_model, x_data, y_data, window_size, target_layer_name)

        for sample_idx in range(len(x_data)):
            # real sample length
            real_length = l_data[sample_idx] if l_data[sample_idx] < max_length else max_length
            # get the interpretation feature information
            if y_data[sample_idx] == y_prediction[(batch_idx * len(x_data)) + sample_idx]:
                for j, w in enumerate(interpretation_heatmap[sample_idx][:real_length]):
                    if w > 0 and x_data[sample_idx][j] in vocab_dict:
                        if_dict.setdefault(y_data[sample_idx], dict())
                        if_dict[y_data[sample_idx]][vocab_dict[x_data[sample_idx][j]]] = if_dict[y_data[sample_idx]].setdefault(vocab_dict[x_data[sample_idx][j]], 0) + w
                        if_count.setdefault(y_data[sample_idx], dict())
                        if_count[y_data[sample_idx]][vocab_dict[x_data[sample_idx][j]]] = if_count[y_data[sample_idx]].setdefault(vocab_dict[x_data[sample_idx][j]], 0) + 1

    print('Generating the interpretation features for all the data is over')
    # get the average importance values of the interpretation_features
    for category, sub_dict in if_dict.items():
        for word, value in sub_dict.items():
            if_average.setdefault(category, dict())
            if_average[category][word] = np.divide(value, if_count[category][word])

    post_processed_if_dict = post_processing(if_average, if_count)

    with open(settings.post_processed_IF, 'wb') as f:
        pkl.dump(post_processed_if_dict, f, pkl.HIGHEST_PROTOCOL)
    with open(settings.IF, 'wb') as f:
        pkl.dump(if_dict, f, pkl.HIGHEST_PROTOCOL)
    with open(settings.count_IF, 'wb') as f:
        pkl.dump(if_count, f, pkl.HIGHEST_PROTOCOL)
    with open(settings.average_IF, 'wb') as f:
        pkl.dump(if_average, f, pkl.HIGHEST_PROTOCOL)


# Tested
def post_processing(if_average, if_count, alpha_threshold=0.3, beta_threshold=0.25, print_if_dict=False):
    """
    """
    # final dictionary
    post_processed_if_dict = dict()

    for category, sub_dict in if_average.items():
        for word, value in sub_dict.items():
            if value >= alpha_threshold and if_count[category][word] > 10 and len(word) >= 2:
                is_true = True
                for category_idx in range(len(if_average)):
                    if category_idx in if_average and category_idx != category:
                        if word not in if_average[category_idx] or value >= if_average[category_idx][word] + beta_threshold:
                            continue
                        else:
                            is_true = False
                            break
                if is_true:
                    post_processed_if_dict.setdefault(category, dict())
                    post_processed_if_dict[category][word] = value

    print(f'the length of the interpretation_features dict for all categories: {len(post_processed_if_dict)}')

    if print_if_dict:
        for key, value in post_processed_if_dict.items():
            print(f'The interpretation features of class: {key} is :')
            for item in sorted(value.items(), key=itemgetter(1)):
                print('{} : {:.2f}'.format(item[0], item[1]))
            print(f'The length of the dictionary for class {key} is = {len(value)}')
            print()

    return post_processed_if_dict


# Tested
def generate_if_dict_de(vocab_dict, input_model_path, interpretation_method='grad*input'):
    """
    :param vocab_dict: vocabulary dictionary path
    :param input_model_path: the labels for the input data
    :param interpretation_method: {'saliency', 'grad*input', 'intgrad', 'elrp', 'deeplift'}
    """
    with DeepExplain(session=tf.compat.v1.keras.backend.get_session()) as de:

        if_dict = dict()  # the summation of the relevance scores of each interpretation feature in each document
        if_count = dict()  # the count of each interpretation feature
        if_average = dict()  # average relevance score for each interpretation features

        new_model = models.load_model(input_model_path)

        x_train, y_train, l_train, _, _, _, vocab_dict = train_test_split(settings.data_source,
                                                                          settings.test_split,
                                                                          settings.sequence_length,
                                                                          vocabulary=vocab_dict)

        # convert the index labels to one-hot format
        ys = np.zeros(shape=(len(y_train), settings.num_classes))
        for i, label in enumerate(y_train):
            ys[i, label] = 1

        # shape [None, 50, 128]
        embedding_tensor = new_model.layers[1].output

        # Get tensor before the final activation
        # shape [batch_size, num_classes]
        logits_tensor = new_model.layers[-2].output

        for batch_idx in range(0, len(x_train) // settings.batch_size, settings.batch_size):

            next_batch = batch_idx + settings.batch_size

            x_train_batch = x_train[batch_idx:next_batch]
            y_train_batch = y_train[batch_idx:next_batch]

            y_prediction = new_model.predict(x_train_batch)
            y_prediction = np.argmax(y_prediction, axis=1)

            # Interpretation ===========================================================================================

            # Evaluate the embedding tensor on the model input

            embedding_function = K.function([new_model.input], [embedding_tensor])
            embedding_output = embedding_function([x_train_batch])

            # Run DeepExplain with the embedding as input
            heat_map = de.explain(interpretation_method, logits_tensor * ys[batch_idx:next_batch], embedding_tensor, embedding_output[0])

            # sum the values for the embedding dimension to get the relevance value of each word
            interpretation_heatmap = np.sum(heat_map, axis=-1)
            interpretation_heatmap = np.maximum(interpretation_heatmap, 0)

            for sample_idx in range(settings.batch_size):
                words = [vocab_dict[w] if w in vocab_dict else '<PAD/>' for w in x_train_batch[sample_idx][:l_train[sample_idx]]]
                sample_heatmap = np.round(interpretation_heatmap[sample_idx][:l_train[sample_idx]]/(max(interpretation_heatmap[sample_idx][:l_train[sample_idx]]) + 0.00001), 2)

                # get the interpretation feature information
                if y_train_batch[sample_idx] == y_prediction[sample_idx]:
                    for j, w in enumerate(sample_heatmap):
                        if w > 0:
                            if_dict.setdefault(y_train_batch[sample_idx], dict())
                            if_dict[y_train_batch[sample_idx]][words[j]] = if_dict[y_train_batch[sample_idx]].setdefault(words[j], 0) + w
                            if_count.setdefault(y_train_batch[sample_idx], dict())
                            if_count[y_train_batch[sample_idx]][words[j]] = if_count[y_train_batch[sample_idx]].setdefault(words[j], 0) + 1

        # get the average importance values of the interpretation_features
        for category, sub_dict in if_dict.items():
            for word, value in sub_dict.items():
                if_average.setdefault(category, dict())
                if_average[category][word] = np.divide(value, if_count[category][word])

        post_processed_if_dict = post_processing(if_average, if_count, alpha_threshold=0.2)

        with open(settings.post_processed_IF, 'wb') as f:
            pkl.dump(post_processed_if_dict, f, pkl.HIGHEST_PROTOCOL)
        with open(settings.IF, 'wb') as f:
            pkl.dump(if_dict, f, pkl.HIGHEST_PROTOCOL)
        with open(settings.count_IF, 'wb') as f:
            pkl.dump(if_count, f, pkl.HIGHEST_PROTOCOL)
        with open(settings.average_IF, 'wb') as f:
            pkl.dump(if_average, f, pkl.HIGHEST_PROTOCOL)


# ======================================================================================================================
# Running the code
generate_if_dict(vocab_dict=settings.vocabulary_dict,
                 input_model_path=settings.model_path,
                 window_size=settings.filter_sizes,
                 target_layer_name=settings.target_layer_name,
                 max_length=settings.sequence_length)


# generate_if_dict_de(vocab_dict=settings.vocabulary_dict, input_model_path=settings.model_path, interpretation_method='grad*input')
