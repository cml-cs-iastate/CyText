import os
import csv
import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.contrib import learn

from src.SELFIE.clf import data_helper
from src.SELFIE.clf.utils import generate_ground_truth, random_remove_words, uniform_remove_word, average_drop, increase_confidence

# Show warnings and errors only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# File paths
tf.flags.DEFINE_string('test_data_file', None, 'Test data file path')
tf.flags.DEFINE_string('run_dir', None, 'Restore the model from this run')
tf.flags.DEFINE_string('checkpoint', None, 'Restore the graphs from this checkpoint')
tf.flags.DEFINE_string('if_dictionary_path', 'C:/Users/moham/Documents/Projects/DeepModelInterpretation/src/SELFIE/clf/runs/CNN_Amazon_CAM_ReLU/interpretation_features_200k/final_IF.pkl', 'The path to the interpretation features dictionary')

# Model choices
tf.flags.DEFINE_boolean('generate_IF', False, 'generate and save the interpretation_features of the prediction')
tf.flags.DEFINE_boolean('selfie', True, 'run the classification using SELFIE')
tf.flags.DEFINE_boolean('salience_map', False, 'build the classifier for Salience Map')
tf.flags.DEFINE_boolean('lrp', False, 'run LRP')
tf.flags.DEFINE_boolean('remove_words', True, 'remove words from the inputs data')

# Test batch size
tf.flags.DEFINE_integer('batch_size', 32, 'Test batch size')

FLAGS = tf.app.flags.FLAGS

# Restore parameters
with open(os.path.join(FLAGS.run_dir, 'params.pkl'), 'rb') as f:
    params = pkl.load(f, encoding='bytes')

# Restore vocabulary processor
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(os.path.join(FLAGS.run_dir, 'vocab'))
vocab_dict = vocab_processor.vocabulary_._mapping
inv_vocab_dict = dict(zip(vocab_dict.values(), vocab_dict.keys()))

# Load test data
data, labels, lengths, _, _ = data_helper.load_data(file_path=FLAGS.test_data_file,
                                                    min_frequency=params['min_frequency'],
                                                    max_length=params['max_length'],
                                                    vocab_processor=vocab_processor,
                                                    shuffle=False)

# load the true feature interpretation dictionary
if FLAGS.if_dictionary_path is not None:
    with open(FLAGS.if_dictionary_path, 'rb') as f:
        ground_truth_dict = pkl.load(f)

# Restore graphs
graph = tf.Graph()
with graph.as_default():
    sess = tf.compat.v1.Session()
    # Restore meta-graphs
    saver = tf.compat.v1.train.import_meta_graph('{}.meta'.format(os.path.join(FLAGS.run_dir, 'model', FLAGS.checkpoint)))
    # Restore weights
    saver.restore(sess, os.path.join(FLAGS.run_dir, 'model', FLAGS.checkpoint))

    # get the name of all the tensors
    # print([n.name for n in tf.get_default_graph().as_graph_def().node])

    # Get tensors
    input_x = graph.get_tensor_by_name('input_x:0')
    input_y = graph.get_tensor_by_name('input_y:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    predictions = graph.get_tensor_by_name('softmax/predictions:0')
    logits = graph.get_tensor_by_name('softmax/softmax_logits:0')
    accuracy = graph.get_tensor_by_name('accuracy/accuracy:0')
    normalized_heat_map = graph.get_tensor_by_name('interpretation/normalized_heat_map:0')
    interpretation_features = graph.get_tensor_by_name('interpretation_placeholders:0')

    # Generate batches
    batches = data_helper.batch_iter(data, labels, lengths, FLAGS.batch_size, 1)

    num_batches = int(len(data) / FLAGS.batch_size)
    all_predictions = []
    sum_accuracy = 0

    if FLAGS.selfie or FLAGS.salience_map or FLAGS.lrp:
        precision = 0
        recall = 0
        kappa = 0
        heat_map = list()            # the heat map of a given input to reflect the importance of each word

    if FLAGS.generate_IF:
        IF = dict()                  # the summation of the interpretation value for each interpretation feature
        count_IF = dict()            # the count of each interpretation feature weighted by the interpretation method
        average_IF = dict()          # averaging the true interpretation_features per dataset

    final_logits_ = list()
    final_predictions = list()

    if FLAGS.remove_words:
        with open(os.path.join(FLAGS.run_dir, 'heat_map.pkl'), 'rb') as f:
            loaded_heat_map = pkl.load(f)

    # Testing
    print('Testing started')
    for counter, batch in enumerate(batches):
        x_test, y_test, x_lengths = batch

        # real sample length
        real_lengths = [l if l < params['max_length'] else params['max_length'] for l in x_lengths]

        if FLAGS.remove_words:
            # remove words from input
            # x_test = random_remove_words(x_test, real_lengths, remove_ratio=0.5)
            x_test = uniform_remove_word(x_test, real_lengths, remove_ratio=0.5, heat_map_=loaded_heat_map[counter * FLAGS.batch_size:(counter + 1) * FLAGS.batch_size], most_relevant_=False)

        feed_dict = {input_x: x_test, input_y: y_test, keep_prob: 1.0}

        if not FLAGS.selfie and not FLAGS.salience_map and not FLAGS.lrp:
            batch_predictions, batch_accuracy, logits_ = sess.run([predictions, accuracy, logits], feed_dict)
            final_logits_.append(logits_)
            final_predictions.append(batch_predictions)
        else:
            batch_predictions, batch_accuracy, logits_, normalized_heat_map_ = sess.run([predictions, accuracy, logits, normalized_heat_map], feed_dict)
            final_logits_.append(logits_)
            final_predictions.append(batch_predictions)

            if not FLAGS.remove_words and FLAGS.if_dictionary_path is not None:
                # calculate the Kappa, Precision, and Recall
                interpretation_ground_truth = generate_ground_truth(x_test, y_test, inv_vocab_dict, ground_truth_dict)
                for sample_idx in range(FLAGS.batch_size):
                    if batch_predictions[sample_idx] == y_test[sample_idx]:
                        a, b, c, d = 0, 0, 0, 0
                        for word_idx in range(real_lengths[sample_idx]):
                            if interpretation_ground_truth[sample_idx][word_idx] == 1 and normalized_heat_map_[sample_idx][word_idx] >= 0.5:
                                a += 1  # TP
                            elif interpretation_ground_truth[sample_idx][word_idx] == 1 and normalized_heat_map_[sample_idx][word_idx] < 0.5:
                                b += 1  # FN
                            elif interpretation_ground_truth[sample_idx][word_idx] == 0 and normalized_heat_map_[sample_idx][word_idx] >= 0.5:
                                c += 1  # FP
                            elif interpretation_ground_truth[sample_idx][word_idx] == 0 and normalized_heat_map_[sample_idx][word_idx] < 0.5:
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
                    # heat-map
                    heat_map.append(normalized_heat_map_[sample_idx][:real_lengths[sample_idx]])
            else:
                for sample_idx in range(FLAGS.batch_size):
                    heat_map.append(normalized_heat_map_[sample_idx][:real_lengths[sample_idx]])

        if FLAGS.generate_IF:
            # store the predicted interpretation_features information per input sample
            for sample_idx in range(FLAGS.batch_size):
                # get the words from the embedding
                words = [inv_vocab_dict[w] for w in x_test[sample_idx][:real_lengths[sample_idx]]]
                # get the interpretation feature information
                if batch_predictions[sample_idx] == y_test[sample_idx]:
                    for j, w in enumerate(normalized_heat_map_[sample_idx][:real_lengths[sample_idx]]):
                        if w > 0:
                            IF.setdefault(batch_predictions[sample_idx], dict())
                            IF[batch_predictions[sample_idx]][words[j]] = IF[batch_predictions[sample_idx]].setdefault(words[j], 0) + w
                            count_IF.setdefault(batch_predictions[sample_idx], dict())
                            count_IF[batch_predictions[sample_idx]][words[j]] = count_IF[batch_predictions[sample_idx]].setdefault(words[j], 0) + 1

        sum_accuracy += batch_accuracy
        all_predictions = np.concatenate([all_predictions, batch_predictions])

    # print the results
    print(f'Test accuracy: {sum_accuracy / num_batches}')
    if not FLAGS.remove_words and (FLAGS.selfie or FLAGS.salience_map or FLAGS.lrp):
        precision /= ((counter + 1) * FLAGS.batch_size)
        recall /= ((counter + 1) * FLAGS.batch_size)
        kappa /= ((counter + 1) * FLAGS.batch_size)
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'Kappa: {kappa}')

    # store all predictions
    with open(os.path.join(FLAGS.run_dir, 'predictions.csv'), 'w', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['True class', 'Prediction'])
        for sample_idx in range(len(all_predictions)):
            csv_writer.writerow([labels[sample_idx], all_predictions[sample_idx]])
        print('Predictions saved to {}'.format(os.path.join(FLAGS.run_dir, 'predictions.csv')))

    # store interpretation features statistics to files
    if FLAGS.generate_IF:
        # get the average importance values of the interpretation_features
        for category, sub_dict in IF.items():
            for word, value in sub_dict.items():
                average_IF.setdefault(category, dict())
                average_IF[category][word] = np.divide(value, count_IF[category][word])

        with open(os.path.join(FLAGS.run_dir, 'IF.pkl'), 'wb') as f:
            pkl.dump(IF, f, pkl.HIGHEST_PROTOCOL)

        with open(os.path.join(FLAGS.run_dir, 'count_IF.pkl'), 'wb') as f:
            pkl.dump(count_IF, f, pkl.HIGHEST_PROTOCOL)

        with open(os.path.join(FLAGS.run_dir, 'average_IF.pkl'), 'wb') as f:
            pkl.dump(average_IF, f, pkl.HIGHEST_PROTOCOL)

    if not FLAGS.remove_words:
        # store heat map
        with open(os.path.join(FLAGS.run_dir, 'heat_map.pkl'), 'wb') as f:
            pkl.dump(heat_map, f, pkl.HIGHEST_PROTOCOL)

        with open(os.path.join(FLAGS.run_dir, 'predictions.pkl'), 'wb') as f:
            pkl.dump(final_predictions, f, pkl.HIGHEST_PROTOCOL)
        with open(os.path.join(FLAGS.run_dir, 'logits.pkl'), 'wb') as f:
            pkl.dump(final_logits_, f, pkl.HIGHEST_PROTOCOL)
    else:
        with open(os.path.join(FLAGS.run_dir, 'predictions.pkl'), 'rb') as f:
            original_predictions = pkl.load(f)
        with open(os.path.join(FLAGS.run_dir, 'logits.pkl'), 'rb') as f:
            original_logits_ = pkl.load(f)

        print(f'Average Drop: {average_drop(original_logits_, final_logits_, original_predictions, FLAGS.batch_size)}')
        print(f'Increase in confidence: {increase_confidence(original_logits_, final_logits_, original_predictions, FLAGS.batch_size)}')
