import numpy as np
from tensorflow.keras import backend as K


# Tested
def grad_cam_single(input_model, input_x, input_y, target_layer_name):
    # convert input_y from one-hot format into index format
    input_y = np.argmax(input_y)

    output_y = input_model.output[0, input_y]

    conv_output = input_model.get_layer(target_layer_name).output

    # Calculate the gradients from the target class to the target layer
    grads = K.gradients(output_y, conv_output)[0]

    gradient_function = K.function([input_model.input], [conv_output, grads])

    # shape output (batch_size, sequence_length, num_filters)
    # shape grads_val (batch_size, sequence_length, num_filters)
    output, grads_val = gradient_function(K.expand_dims(input_x, axis=0))

    output, grads_val = output[0, :, :], grads_val[0, :, :]

    # get the average weights of the entire filter
    # shape (num_filters)
    weights = np.mean(grads_val, axis=0)

    # shape [sequence_length]
    cam = np.dot(output, weights)

    # Distance from the mean
    cam = cam - cam.mean()

    # Keep only positive values
    cam = np.maximum(cam, 0)

    # Normalize
    cam = cam / cam.max() if cam.max() != 0 else cam

    return cam


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

    for _ in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        shuffled_labels = labels[shuffle_indices]
        shuffled_lengths = lengths[shuffle_indices]
        for batch_num in range(epoch_length):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)

            xdata = shuffled_data[start_index: end_index]
            ydata = shuffled_labels[start_index: end_index]
            sequence_length = shuffled_lengths[start_index: end_index]

            yield xdata, ydata, sequence_length


def load_data_npy(x_text_path, y_text_path, length_text_path, vocab_path, sequence_length):
    """
    Loads and preprocessed data.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    with open(x_text_path, 'r') as f:
        ff = f.readlines()
    sentences = list()
    for line in ff:
        sentences.append(line.strip().split()[:-1])
    x = pad_sentences(sentences, sequence_length, padding_word='0')
    x = np.array(x, dtype=int)

    y = np.loadtxt(y_text_path, dtype=int)
    lengths = np.loadtxt(length_text_path, dtype=int)

    # load vocabulary
    with open(vocab_path, 'rb') as f:
        vocabulary = pkl.load(f)

    return [x, y, lengths, vocabulary]


# Tested
def perturb_input(if_dict, vocab_dict_path, max_length, random=False, most_relevant=False, remove_ratio=0.5):
    """
    """
    with open(if_dict, 'rb') as f:
        if_dict = pkl.load(f)

    x_perturb = []

    x_train, y_train, l_train, _, _, _, vocab_dict = train_test_split(settings.data_source,
                                                                      settings.test_split,
                                                                      settings.sequence_length,
                                                                      vocabulary=vocab_dict_path)

    batches = batch_iter(x_train, y_train, l_train, settings.batch_size, num_epochs=1)

    for batch in batches:
        x_data, y_data, l_data = batch

        # real sample length
        real_length = [l if l < max_length else max_length for l in l_data]

        # mark the interpretation ground truth for the document words
        interpretation_ground_truth = generate_ground_truth(x_data, y_data, vocab_dict, if_dict, binary=False)

        # remove words from input
        if random:
            t = random_remove_words(x_data, real_length, remove_ratio)
        else:
            t = uniform_remove_word(x_data,
                                    real_length,
                                    remove_ratio,
                                    heat_map=interpretation_ground_truth,
                                    most_relevant=most_relevant)

        x_perturb.append(t)

    x_perturb = np.concatenate(x_perturb, axis=0)

    return x_perturb