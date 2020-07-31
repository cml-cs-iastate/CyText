from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding, Concatenate, Activation


def cnn_classifier(num_classes, dropout, embedding_dim, num_filters, filter_sizes, sequence_length, fully_connected_dim, vocabulary_inv, combined_loss=False):
    """
    CNN classification model for sentiment analysis based on "Convolutional Neural Networks for Sentence Classification"
     by Yoon Kim
    """

    input_shape = (sequence_length,)
    model_input = Input(shape=input_shape, name='input')

    x = Embedding(len(vocabulary_inv), embedding_dim, input_length=sequence_length, name='embedding')(model_input)

    x = Dropout(dropout)(x)

    # Convolutional block
    conv_blocks = []

    for sz in filter_sizes:
        conv = Convolution1D(filters=num_filters,
                             kernel_size=sz,
                             padding='valid',  # 'same' is better but we are not sure if the padding is only going to be at the end of the text
                             activation='relu',
                             strides=1)(x)
        conv = MaxPooling1D(pool_size=sequence_length - sz + 1)(conv)
        conv = Flatten()(conv)
        conv_blocks.append(conv)

    # Concatenate the convolutional layers
    x = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

    x = Dropout(dropout)(x)
    x = Dense(fully_connected_dim, activation='relu')(x)

    logits = Dense(num_classes, activation='linear', name='logits')(x)
    model_output = Activation('softmax', name='predictions')(logits)

    model = Model(model_input, model_output)

    if combined_loss:
        return model, model_input, logits, conv_blocks

    return model
