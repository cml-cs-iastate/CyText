import tensorflow as tf


class cnn_clf(object):
    """
    A CNN classifier for text classification
    """
    def __init__(self, config):
        """
        the initiation function for the class
        :param config: the object that holds all the argpars from the user.
        https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras
        """
        # if True, the classifier will use the combined loss function and Jaccard accuracy
        self.selfie = config.selfie
        self.salience_map = config.salience_map
        self.lrp = config.lrp
        self.attention = config.attention
        self.slide_activation_selfie = False

        self.batch_size = config.batch_size
        self.max_length = config.max_length
        self.num_classes = config.num_classes
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        self.filter_sizes = list(map(int, config.filter_sizes.split(",")))
        self.num_filters = config.num_filters
        self.l2_reg_lambda = config.l2_reg_lambda

        # Placeholders
        self.input_x = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, self.max_length], name='input_x')
        self.input_y = tf.compat.v1.placeholder(dtype=tf.int64, shape=[None], name='input_y')
        self.keep_prob = tf.compat.v1.placeholder(dtype=tf.float32, name='keep_prob')
        self.true_features = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.max_length], name='interpretation_placeholders')

        # L2 loss
        self.l2_loss = tf.constant(0.0)

        num_filters_total = self.num_filters * len(self.filter_sizes)
        print(f'num_filters_total: {num_filters_total}\n')

        # word embedding
        with tf.name_scope('embedding'):
            embedding = tf.Variable(tf.random.uniform([self.vocab_size, self.embedding_size], -1.0, 1.0), name='embedding')
            embed = tf.nn.embedding_lookup(embedding, self.input_x, name='embed')
            # shape = (?, 250, 300, 1) [batch_size, max_length, embedding_size, 1]
            self.inputs = tf.expand_dims(embed, -1, name='inputs')

        # convolution & max-pooling
        with tf.name_scope('convolution'):
            convolution_layer_2 = self.convolution_layer(filter_size=2)
            convolution_layer_3 = self.convolution_layer(filter_size=3)
            convolution_layer_4 = self.convolution_layer(filter_size=4)

            pooled_output = list()
            pooled_output.append(convolution_layer_2[1])
            pooled_output.append(convolution_layer_3[1])
            pooled_output.append(convolution_layer_4[1])
            # shape=(?, 384) [batch_size, num_filters_total]
            h_pool = tf.squeeze(tf.concat(pooled_output, 3), [1, 2], name='concat_pooled_output')

            # dropout  # shape=(?, 384) [batch_size, num_filters_total]
            h_drop = tf.nn.dropout(h_pool, keep_prob=self.keep_prob)

        if self.attention:
            with tf.name_scope('attention'):
                # attention weights and bias
                w_attention = tf.Variable(tf.random.truncated_normal([num_filters_total, num_filters_total], stddev=0.1), name='attention_w')
                b_attention = tf.Variable(tf.constant(0.1, shape=[num_filters_total]), name='attention_b')

                # fully connected layer shape=(?, 384) [batch_size, num_filters_total]
                attention_representation = tf.nn.softmax(tf.nn.relu(tf.matmul(h_drop, w_attention) + b_attention))

                # shape=(?, 384) [batch_size, num_filters_total]
                h_drop = tf.identity(tf.multiply(attention_representation, h_drop), name='attention_representation')

        # softmax
        with tf.name_scope('softmax'):
            # softmax weights and bias
            softmax_w = tf.Variable(tf.random.truncated_normal([num_filters_total, self.num_classes], stddev=0.1), name='softmax_w')
            softmax_b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name='softmax_b')

            # Add L2 regularization to output layer
            self.l2_loss += tf.nn.l2_loss(softmax_w)
            self.l2_loss += tf.nn.l2_loss(softmax_b)

            self.logits = tf.matmul(h_drop, softmax_w) + softmax_b
            self.softmax_logits = tf.nn.softmax(self.logits, name='softmax_logits')
            self.predictions = tf.argmax(self.softmax_logits, 1, name='predictions')

        # interpretation: SELFIE, Salience Map, or LRP
        with tf.name_scope('interpretation'):
            # get the output neuron corresponding to the class of interest (predicted label_id)
            # shape = (?,) [batch_size]
            target_logits = tf.reduce_max(input_tensor=self.logits, axis=1, name='target_logits')

            if self.lrp:
                self.heat_map = tf.identity(self.run_lrp(softmax_w, softmax_b, h_drop, target_logits, num_filters_total, convolution_layer_2, convolution_layer_3, convolution_layer_4), name='normalized_heat_map')
            elif self.attention:
                self.heat_map = tf.identity(self.run_attention(h_drop, convolution_layer_2, convolution_layer_3, convolution_layer_4), name='normalized_heat_map')
            else:
                # get last convolutional layers gradients
                # shape = (?, 249, 1, 128) [batch_size, max_length-filter_size+1, 1, num_filters]
                h_2_gradients = tf.gradients(target_logits, convolution_layer_2[0], name='gradients_h_2')[0]
                h_3_gradients = tf.gradients(target_logits, convolution_layer_3[0], name='gradients_h_3')[0]
                h_4_gradients = tf.gradients(target_logits, convolution_layer_4[0], name='gradients_h_4')[0]
                if self.salience_map:
                    # generate the salience map for each convolutional layer
                    heat_map_2 = self.run_salience_map(h_2_gradients, 2)  # shape=(?, 250)
                    heat_map_3 = self.run_salience_map(h_3_gradients, 3)  # shape=(?, 250)
                    heat_map_4 = self.run_salience_map(h_4_gradients, 4)  # shape=(?, 250)
                else:
                    # generate the heat map for each convolutional layer
                    heat_map_2 = self.run_selfie(h_2_gradients, convolution_layer_2[0], convolution_layer_2[2], convolution_layer_2[3], 2, 'heat_map_2')  # shape=(?, 250)
                    heat_map_3 = self.run_selfie(h_3_gradients, convolution_layer_3[0], convolution_layer_3[2], convolution_layer_3[3], 3, 'heat_map_3')  # shape=(?, 250)
                    heat_map_4 = self.run_selfie(h_4_gradients, convolution_layer_4[0], convolution_layer_4[2], convolution_layer_4[3], 4, 'heat_map_4')  # shape=(?, 250)

                # combine the three convolutional layers
                # shape=(32, 250)
                heat_map = tf.add_n([heat_map_2, heat_map_3, heat_map_4], name='heat_map')
                # normalize [0:1]  # shape=(32, 250)
                max_value_tiled = tf.tile(tf.reshape(tf.reduce_max(heat_map, axis=1), [self.batch_size, 1]), [1, self.max_length])
                # shape=(32, 250)
                self.heat_map = tf.divide(heat_map, max_value_tiled + tf.constant(1e-10), name='normalized_heat_map')

        # Loss
        with tf.name_scope('loss'):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits, name='losses')
            # Calculate Jaccard Similarity
            intersection = tf.reduce_sum(self.true_features * self.heat_map, axis=1)
            union = tf.reduce_sum(self.true_features + self.heat_map, axis=1)
            jaccard_predictions = tf.divide(intersection, (union - intersection) + tf.constant(1e-10))

            self.jaccard_loss = tf.reduce_mean(tf.subtract(1.0, jaccard_predictions, name='jaccard_loss'))

            self.jaccard_accuracy = tf.reduce_mean(tf.cast(jaccard_predictions, tf.float32), name='jaccard_accuracy')

            if not self.selfie:
                self.cost = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss
            else:
                # Interpretation accuracy
                self.cost = tf.reduce_mean(losses) + self.jaccard_loss + self.l2_reg_lambda * self.l2_loss

        # Accuracy
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, self.input_y)
            self.correct_num = tf.reduce_sum(tf.cast(correct_predictions, tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')

    def convolution_layer(self, filter_size):
        """
        convolution layer creator
        :param filter_size: the filter window size
        :type filter_size: int
        :return:
        """
        with tf.compat.v1.variable_scope('convolution_layer_%s' % filter_size):
            # convolution
            filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
            w = tf.compat.v1.get_variable('weights', filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
            b = tf.compat.v1.get_variable('biases', [self.num_filters], initializer=tf.constant_initializer(0.0))

            # shape = (?, 249, 1, 128) [batch_size, max_length-filter_size+1, 1, num_filters]
            convolution = tf.nn.conv2d(self.inputs, w, strides=[1, 1, 1, 1], padding='VALID', name='conv')

            # activation function
            # shape = (?, 249, 1, 128) [batch_size, max_length-filter_size+1, 1, num_filters]
            h = tf.nn.relu(tf.nn.bias_add(convolution, b), name='relu')

            # max-pooling
            # shape = (?, 1, 1, 128) [batch_size, 1, 1, num_filters]
            pooled = tf.nn.max_pool(h,
                                    ksize=[1, self.max_length - filter_size + 1, 1, 1],
                                    strides=[1, 1, 1, 1],
                                    padding='VALID',
                                    name='pool')

            return [h, pooled, w, b]

    def run_salience_map(self, h_gradients, filter_size):
        """
        generate the salience map
        :param h_gradients: the gradient to the convolutional layer
        :param filter_size: the filter window sizes
        :return:
        """
        # shape = (32, 250) [batch_size, max_length-filter_size+1]
        alpha = tf.concat([tf.reduce_sum(tf.squeeze(tf.nn.relu(h_gradients), [2]), axis=2), tf.zeros([self.batch_size, filter_size - 1])], axis=1)

        if filter_size == 2:
            # shape = (32, 250)
            z = alpha + tf.concat([tf.zeros([self.batch_size, 1]), alpha[:, :-1]], axis=1)
        elif filter_size == 3:
            z = alpha + tf.concat([tf.zeros([self.batch_size, 1]), alpha[:, :-1]], axis=1) + \
                        tf.concat([tf.zeros([self.batch_size, 2]), alpha[:, :-2]], axis=1)
        elif filter_size == 4:
            z = alpha + tf.concat([tf.zeros([self.batch_size, 1]), alpha[:, :-1]], axis=1) + \
                            tf.concat([tf.zeros([self.batch_size, 2]), alpha[:, :-2]], axis=1) + \
                            tf.concat([tf.zeros([self.batch_size, 3]), alpha[:, :-3]], axis=1)
        return z

    def run_selfie(self, h_gradients, h_, w_, b_, filter_size, tensor_name):
        """
        generate the class activation map
        :param h_gradients: the gradient to the convolutoinal layer
        :param h_: the output of the convolutional layer
        :param w_: the kernel of filter size filter_size
        :param b_: the kernel bias
        :param filter_size: the filter window sizes
        :param tensor_name: the name of the output tensor
        :return:
        """
        if self.slide_activation_selfie:
            # shape = (32, 250, 1, 128) [batch_size, max_length-filter_size+1, 1, num_filters]
            z = tf.nn.relu(tf.concat([h_gradients, tf.zeros([self.batch_size, filter_size - 1, 1, self.num_filters])], axis=1))

            if filter_size == 2:
                # shape=(32, 250, 1, 128)
                z_0 = tf.nn.bias_add(tf.nn.conv2d(self.inputs, tf.reshape(w_[0, :, :, :], [1, self.embedding_size, 1, self.num_filters]), strides=[1, 1, 1, 1], padding='VALID'), tf.divide(b_, filter_size))
                z_final = tf.multiply(z_0, z)
                z_1 = tf.nn.bias_add(tf.nn.conv2d(self.inputs, tf.reshape(w_[1, :, :, :], [1, self.embedding_size, 1, self.num_filters]), strides=[1, 1, 1, 1], padding='VALID'), tf.divide(b_, filter_size))
                z_final += tf.multiply(z_1, tf.concat([tf.zeros([self.batch_size, 1, 1, self.num_filters]), z[:, :-1, :, :]], axis=1))
            elif filter_size == 3:
                # shape=(32, 250, 1, 128)
                z_0 = tf.nn.bias_add(tf.nn.conv2d(self.inputs, tf.reshape(w_[0, :, :, :], [1, self.embedding_size, 1, self.num_filters]), strides=[1, 1, 1, 1], padding='VALID'), tf.divide(b_, filter_size))
                z_final = tf.multiply(z_0, z)
                z_1 = tf.nn.bias_add(tf.nn.conv2d(self.inputs, tf.reshape(w_[1, :, :, :], [1, self.embedding_size, 1, self.num_filters]), strides=[1, 1, 1, 1], padding='VALID'), tf.divide(b_, filter_size))
                z_final += tf.multiply(z_1, tf.concat([tf.zeros([self.batch_size, 1, 1, self.num_filters]), z[:, :-1, :, :]], axis=1))
                z_2 = tf.nn.bias_add(tf.nn.conv2d(self.inputs, tf.reshape(w_[2, :, :, :], [1, self.embedding_size, 1, self.num_filters]), strides=[1, 1, 1, 1], padding='VALID'), tf.divide(b_, filter_size))
                z_final += tf.multiply(z_2, tf.concat([tf.zeros([self.batch_size, 2, 1, self.num_filters]), z[:, :-2, :, :]], axis=1))
            elif filter_size == 4:
                # shape=(32, 250, 1, 128)
                z_0 = tf.nn.bias_add(tf.nn.conv2d(self.inputs, tf.reshape(w_[0, :, :, :], [1, self.embedding_size, 1, self.num_filters]), strides=[1, 1, 1, 1], padding='VALID'), tf.divide(b_, filter_size))
                z_final = tf.multiply(z_0, z)
                z_1 = tf.nn.bias_add(tf.nn.conv2d(self.inputs, tf.reshape(w_[1, :, :, :], [1, self.embedding_size, 1, self.num_filters]), strides=[1, 1, 1, 1], padding='VALID'), tf.divide(b_, filter_size))
                z_final += tf.multiply(z_1, tf.concat([tf.zeros([self.batch_size, 1, 1, self.num_filters]), z[:, :-1, :, :]], axis=1))
                z_2 = tf.nn.bias_add(tf.nn.conv2d(self.inputs, tf.reshape(w_[2, :, :, :], [1, self.embedding_size, 1, self.num_filters]), strides=[1, 1, 1, 1], padding='VALID'), tf.divide(b_, filter_size))
                z_final += tf.multiply(z_2, tf.concat([tf.zeros([self.batch_size, 2, 1, self.num_filters]), z[:, :-2, :, :]], axis=1))
                z_3 = tf.nn.bias_add(tf.nn.conv2d(self.inputs, tf.reshape(w_[3, :, :, :], [1, self.embedding_size, 1, self.num_filters]), strides=[1, 1, 1, 1], padding='VALID'), tf.divide(b_, filter_size))
                z_final += tf.multiply(z_3, tf.concat([tf.zeros([self.batch_size, 3, 1, self.num_filters]), z[:, :-3, :, :]], axis=1))
        else:
            # shape = (32, 249, 1, 128)
            # alpha = tf.nn.relu(tf.multiply(h_, h_gradients))
            # TODO: either apply 217 or [219:224]
            # shape = (32, 128)
            alpha = tf.reduce_mean(h_gradients, axis=1)
            # shape = (32, 249, 1, 128)
            alpha = tf.tile(tf.reshape(alpha, [self.batch_size, 1, 1, self.num_filters]), [1, self.max_length - filter_size + 1, 1, 1])
            # shape = (32, 249, 1, 128)
            alpha = tf.nn.relu(tf.multiply(h_, alpha))

            if filter_size == 2:
                # shape = (32, 249, 1, 128)
                z_final = alpha + \
                          tf.concat([tf.zeros([self.batch_size, 1, 1, self.num_filters]), alpha[:, :-1, :, :]], axis=1)
                # shape = (32, 250, 1, 128)
                z_final = tf.concat([z_final, tf.zeros([self.batch_size, filter_size - 1, 1, self.num_filters])], axis=1)
            elif filter_size == 3:
                # shape = (32, 248, 1, 128)
                z_final = alpha + \
                          tf.concat([tf.zeros([self.batch_size, 1, 1, self.num_filters]), alpha[:, :-1, :, :]], axis=1) + \
                          tf.concat([tf.zeros([self.batch_size, 2, 1, self.num_filters]), alpha[:, :-2, :, :]], axis=1)
                # shape = (32, 250, 1, 128)
                z_final = tf.concat([z_final, tf.zeros([self.batch_size, filter_size - 1, 1, self.num_filters])], axis=1)
            elif filter_size == 4:
                # shape = (32, 247, 1, 128)
                z_final = alpha + \
                          tf.concat([tf.zeros([self.batch_size, 1, 1, self.num_filters]), alpha[:, :-1, :, :]], axis=1) + \
                          tf.concat([tf.zeros([self.batch_size, 2, 1, self.num_filters]), alpha[:, :-2, :, :]], axis=1) + \
                          tf.concat([tf.zeros([self.batch_size, 3, 1, self.num_filters]), alpha[:, :-3, :, :]], axis=1)
                # shape = (32, 250, 1, 128)
                z_final = tf.concat([z_final, tf.zeros([self.batch_size, filter_size - 1, 1, self.num_filters])], axis=1)

        return tf.identity(tf.reduce_sum(tf.squeeze(z_final, [2]), axis=2), tensor_name)

    def run_lrp(self, softmax_w_, softmax_b_, h_drop_, target_logits_, num_filters_total_, convolution_layer_2_, convolution_layer_3_, convolution_layer_4_):
        # reshape the weights  # shape = (32, 2, 300, 128)
        w_2 = tf.tile(tf.reshape(tf.squeeze(convolution_layer_2_[2], [2]), [1, 2, self.embedding_size, self.num_filters]), [self.batch_size, 1, 1, 1])
        w_3 = tf.tile(tf.reshape(tf.squeeze(convolution_layer_3_[2], [2]), [1, 3, self.embedding_size, self.num_filters]), [self.batch_size, 1, 1, 1])
        w_4 = tf.tile(tf.reshape(tf.squeeze(convolution_layer_4_[2], [2]), [1, 4, self.embedding_size, self.num_filters]), [self.batch_size, 1, 1, 1])

        final_heat_map = list()

        for i in range(self.batch_size):
            """LRP of the the last layer to concatenated layer"""
            softmax_weights_ = softmax_w_[:, self.predictions[i]]       # shape=(384,) [num_filters_total]
            softmax_bias_ = softmax_b_[self.predictions[i]]             # shape=()     [1]

            # shape=(384,) [num_filters_total]
            z_j = tf.add(tf.multiply(h_drop_[i], softmax_weights_), tf.divide(softmax_bias_, num_filters_total_), name='z_j')
            # tf.add(softmax_bias_, tf.multiply(0.01, tf.where(target_logits_[i] >= 0, 1.0, -1.0)))

            # shape=(384,) [num_filters_total]
            r_j = tf.multiply(tf.divide(z_j, tf.reduce_sum(z_j, axis=0)), target_logits_[i], name='r_j')

            """LRP of the concatenated layer to input layer"""
            # get the index of the maximum pooling values of each the convolutional layer
            # shape=(128, 1) [num_filters, 1]
            max_pool_index_2 = tf.reshape(tf.argmax(tf.squeeze(convolution_layer_2_[0][i, :, :, :], [1]), 0), [self.num_filters, 1], name='max_pool_index_2')
            max_pool_index_3 = tf.reshape(tf.argmax(tf.squeeze(convolution_layer_3_[0][i, :, :, :], [1]), 0), [self.num_filters, 1], name='max_pool_index_3')
            max_pool_index_4 = tf.reshape(tf.argmax(tf.squeeze(convolution_layer_4_[0][i, :, :, :], [1]), 0), [self.num_filters, 1], name='max_pool_index_4')

            inputs_for_lrp_2 = list()       # shape = (2, 300, 1)
            inputs_for_lrp_3 = list()       # shape = (3, 300, 1)
            inputs_for_lrp_4 = list()       # shape = (4, 300, 1)
            for j in range(self.num_filters):
                inputs_for_lrp_2.append(self.inputs[i, max_pool_index_2[j][0]:max_pool_index_2[j][0] + 2, :, :])
            for j in range(self.num_filters):
                inputs_for_lrp_3.append(self.inputs[i, max_pool_index_3[j][0]:max_pool_index_3[j][0] + 3, :, :])
            for j in range(self.num_filters):
                inputs_for_lrp_4.append(self.inputs[i, max_pool_index_4[j][0]:max_pool_index_4[j][0] + 4, :, :])

            inputs_for_lrp_2 = tf.concat(inputs_for_lrp_2, axis=2)  # shape = (2, 300, 128)
            inputs_for_lrp_3 = tf.concat(inputs_for_lrp_3, axis=2)  # shape = (3, 300, 128)
            inputs_for_lrp_4 = tf.concat(inputs_for_lrp_4, axis=2)  # shape = (4, 300, 128)

            # get the lrp to window size = 2
            # shape=(2, 128) [filter_size, num_filters]
            z_i = tf.add(tf.reduce_sum(tf.multiply(inputs_for_lrp_2, w_2[i, :, :, :]), axis=1), tf.divide(convolution_layer_2_[3], self.num_filters))
            # tf.add(convolution_layer_2_[3], tf.multiply(0.01, tf.where(r_j_2 >= 0, tf.ones_like(r_j_2), -1 * tf.ones_like(r_j_2)))))

            # shape=(2, 128) [filter_size, num_filters]
            r_i = tf.multiply(tf.divide(z_i, tf.tile(tf.reshape(tf.reduce_sum(z_i, axis=0), [1, self.num_filters]), [2, 1])), r_j[0:128])

            # shape = (250,) [max_length], shape(r_i[0, :]) = (128)
            lrp_heat_map = tf.scatter_nd(indices=max_pool_index_2, updates=r_i[0, :],
                                         shape=tf.constant([self.max_length], dtype=tf.int64))
            lrp_heat_map = tf.add(lrp_heat_map, tf.scatter_nd(indices=max_pool_index_2+1, updates=r_i[1, :],
                                                              shape=tf.constant([self.max_length], dtype=tf.int64)))

            # get the lrp to window size = 3
            # shape=(3, 128) [filter_size, num_filters]
            z_i = tf.add(tf.reduce_sum(tf.multiply(inputs_for_lrp_3, w_3[i, :, :, :]), axis=1), tf.divide(convolution_layer_3_[3], self.num_filters))
            # tf.add(convolution_layer_3_[3], tf.multiply(0.01, tf.where(r_j_3 >= 0, tf.ones_like(r_j_3), -1 * tf.ones_like(r_j_3)))))

            # shape=(3, 128) [filter_size, num_filters]
            r_i = tf.multiply(tf.divide(z_i, tf.tile(tf.reshape(tf.reduce_sum(z_i, axis=0), [1, self.num_filters]), [3, 1])), r_j[128:256])

            # shape = (250) [max_length], shape(r_i[0, :]) = (128)
            lrp_heat_map = tf.add(lrp_heat_map, tf.scatter_nd(indices=max_pool_index_3, updates=r_i[0, :],
                                                              shape=tf.constant([self.max_length], dtype=tf.int64)))
            lrp_heat_map = tf.add(lrp_heat_map, tf.scatter_nd(indices=max_pool_index_3+1, updates=r_i[1, :],
                                                              shape=tf.constant([self.max_length], dtype=tf.int64)))
            lrp_heat_map = tf.add(lrp_heat_map, tf.scatter_nd(indices=max_pool_index_3+2, updates=r_i[2, :],
                                                              shape=tf.constant([self.max_length], dtype=tf.int64)))

            # get the lrp to window size = 4
            # shape=(4, 128) [filter_size, num_filters]
            z_i = tf.add(tf.reduce_sum(tf.multiply(inputs_for_lrp_4, w_4[i, :, :, :]), axis=1), tf.divide(convolution_layer_3_[3], self.num_filters))
            # tf.add(convolution_layer_4_[3], tf.multiply(0.01, tf.where(r_j_4 >= 0, tf.ones_like(r_j_4), -1 * tf.ones_like(r_j_4)))))

            # shape=(4, 128) [filter_size, num_filters]
            r_i = tf.multiply(tf.divide(z_i, tf.tile(tf.reshape(tf.reduce_sum(z_i, axis=0), [1, self.num_filters]), [4, 1])), r_j[256:384])

            # shape = (250) [max_length], shape(r_i[0, :]) = (128)
            lrp_heat_map = tf.add(lrp_heat_map, tf.scatter_nd(indices=max_pool_index_4, updates=r_i[0, :],
                                                              shape=tf.constant([self.max_length], dtype=tf.int64)))
            lrp_heat_map = tf.add(lrp_heat_map, tf.scatter_nd(indices=max_pool_index_4+1, updates=r_i[1, :],
                                                              shape=tf.constant([self.max_length], dtype=tf.int64)))
            lrp_heat_map = tf.add(lrp_heat_map, tf.scatter_nd(indices=max_pool_index_4+2, updates=r_i[2, :],
                                                              shape=tf.constant([self.max_length], dtype=tf.int64)))
            lrp_heat_map = tf.add(lrp_heat_map, tf.scatter_nd(indices=max_pool_index_4+3, updates=r_i[3, :],
                                                              shape=tf.constant([self.max_length], dtype=tf.int64)))

            final_heat_map.append(tf.reshape(lrp_heat_map, [1, self.max_length]))

        final_heat_map = tf.concat(final_heat_map, axis=0)

        return final_heat_map

    def run_attention(self, h_drop_, convolution_layer_2_, convolution_layer_3_, convolution_layer_4_):
        final_heat_map = list()
        for i in range(self.batch_size):
            # get the index of the maximum pooling values of each the convolutional layer
            # shape=(128, 1) [num_filters, 1]
            max_pool_index_2 = tf.reshape(tf.argmax(tf.squeeze(convolution_layer_2_[0][i, :, :, :], [1]), 0), [self.num_filters, 1], name='max_pool_index_2')
            max_pool_index_3 = tf.reshape(tf.argmax(tf.squeeze(convolution_layer_3_[0][i, :, :, :], [1]), 0), [self.num_filters, 1], name='max_pool_index_3')
            max_pool_index_4 = tf.reshape(tf.argmax(tf.squeeze(convolution_layer_4_[0][i, :, :, :], [1]), 0), [self.num_filters, 1], name='max_pool_index_4')

            # shape = (250,) [max_length]
            # window size = 2
            heat_map = tf.nn.relu(tf.scatter_nd(indices=max_pool_index_2, updates=h_drop_[i, :128], shape=tf.constant([self.max_length], dtype=tf.int64)))
            heat_map += tf.nn.relu(tf.scatter_nd(indices=max_pool_index_2 + 1, updates=h_drop_[i, :128], shape=tf.constant([self.max_length], dtype=tf.int64)))

            # window size = 3
            heat_map += tf.nn.relu(tf.scatter_nd(indices=max_pool_index_3, updates=h_drop_[i, 128:256], shape=tf.constant([self.max_length], dtype=tf.int64)))
            heat_map += tf.nn.relu(tf.scatter_nd(indices=max_pool_index_3 + 1, updates=h_drop_[i, 128:256], shape=tf.constant([self.max_length], dtype=tf.int64)))
            heat_map += tf.nn.relu(tf.scatter_nd(indices=max_pool_index_3 + 2, updates=h_drop_[i, 128:256], shape=tf.constant([self.max_length], dtype=tf.int64)))

            # window size = 4
            heat_map += tf.nn.relu(tf.scatter_nd(indices=max_pool_index_4, updates=h_drop_[i, 256:384], shape=tf.constant([self.max_length], dtype=tf.int64)))
            heat_map += tf.nn.relu(tf.scatter_nd(indices=max_pool_index_4 + 1, updates=h_drop_[i, 256:384], shape=tf.constant([self.max_length], dtype=tf.int64)))
            heat_map += tf.nn.relu(tf.scatter_nd(indices=max_pool_index_4 + 2, updates=h_drop_[i, 256:384], shape=tf.constant([self.max_length], dtype=tf.int64)))
            heat_map += tf.nn.relu(tf.scatter_nd(indices=max_pool_index_4 + 3, updates=h_drop_[i, 256:384], shape=tf.constant([self.max_length], dtype=tf.int64)))

            final_heat_map.append(tf.reshape(tf.math.divide(heat_map, tf.reduce_max(heat_map, axis=0)), [1, self.max_length]))

        return tf.concat(final_heat_map, axis=0)
