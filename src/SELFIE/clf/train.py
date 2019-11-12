import os
import time
import pickle as pkl
import tensorflow as tf

from src.SELFIE.clf import data_helper
from src.SELFIE.clf.cnn_classifier import cnn_clf
from src.SELFIE.clf.utils import generate_ground_truth, plot_results

from sklearn.model_selection import train_test_split

# Show warnings and errors only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Parameters
# =============================================================================

# Model choices
tf.flags.DEFINE_boolean('selfie', True, 'build the classifier for the combined loss')
tf.flags.DEFINE_boolean('salience_map', False, 'build the classifier for Salience Map')
tf.flags.DEFINE_boolean('lrp', False, 'run LRP')
tf.flags.DEFINE_string('if_dictionary_path', 'C:/Users/moham/Documents/Projects/DeepModelInterpretation/src/SELFIE/clf/runs/CNN_Amazon_CAM_ReLU/interpretation_features_200k/final_IF.pkl',
                       'The path to the interpretation features dictionary')

# Data parameters
tf.flags.DEFINE_string('data_file', None, 'Data file path')
tf.flags.DEFINE_integer('min_frequency', 0, 'Minimal word frequency')
tf.flags.DEFINE_integer('num_classes', 2, 'Number of classes')
tf.flags.DEFINE_integer('max_length', 250, 'Max document length')
tf.flags.DEFINE_integer('vocab_size', 0, 'Vocabulary size')
tf.flags.DEFINE_float('validation_size', 0.2, 'Validation size')

# Model hyper-parameters
tf.flags.DEFINE_integer('embedding_size', 300, 'Word embedding size.')
tf.flags.DEFINE_string('filter_sizes', '2, 3, 4', 'CNN filter sizes.')
tf.flags.DEFINE_integer('num_filters', 128, 'Number of filters per filter size.')
tf.flags.DEFINE_float('keep_prob', 0.5, 'Dropout probability = 1 - keep_prob')
tf.flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate')
tf.flags.DEFINE_float('l2_reg_lambda', 1e-3, 'L2 regularization lambda')

# Training parameters
tf.flags.DEFINE_integer('batch_size', 32, 'Batch size')
tf.flags.DEFINE_integer('num_epochs', 5, 'Number of epochs')
tf.flags.DEFINE_float('decay_rate', 0.7, 'Learning rate decay (0, 1]')
tf.flags.DEFINE_integer('decay_steps', 2500, 'Learning rate decay steps')
tf.flags.DEFINE_integer('evaluate_every_steps', 1000, 'Evaluate the model on validation set after this many steps')
tf.flags.DEFINE_integer('save_every_steps', 1000, 'Save the model after this many steps')
tf.flags.DEFINE_integer('num_checkpoint', 20, 'Number of models to store')

FLAGS = tf.app.flags.FLAGS


# load and save data
# =============================================================================

# Output files directory
output_directory = os.path.abspath(os.path.join(os.path.curdir, 'runs', str(int(time.time()))))
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

data, labels, lengths, vocab_processor, _ = data_helper.load_data(file_path=FLAGS.data_file,
                                                                  min_frequency=FLAGS.min_frequency,
                                                                  max_length=FLAGS.max_length)

# Save vocabulary processor
vocab_processor.save(os.path.join(output_directory, 'vocab'))

vocab_dict = vocab_processor.vocabulary_._mapping
inv_vocab_processor = dict(zip(vocab_dict.values(), vocab_dict.keys()))

if FLAGS.vocab_size == 0:
    FLAGS.vocab_size = len(vocab_processor.vocabulary_._mapping)
FLAGS.max_length = vocab_processor.max_document_length


# save parameters to file
with open(os.path.join(output_directory, 'params.pkl'), 'wb') as f:
    pkl.dump(FLAGS.flag_values_dict(), f, True)


# split the data to training/validation
x_train, x_valid, y_train, y_valid, train_lengths, valid_lengths = train_test_split(data,
                                                                                    labels,
                                                                                    lengths,
                                                                                    test_size=FLAGS.validation_size,
                                                                                    random_state=22)


# batch iterator
train_data = data_helper.batch_iter(x_train, y_train, train_lengths, FLAGS.batch_size, FLAGS.num_epochs)


# Empty the unused data
data = None
labels = None
lengths = None
vocab_processor = None
vocab_dict = None


# load the true feature interpretation dictionary. I should load the true and false dictionary
if FLAGS.selfie:
    with open(FLAGS.if_dictionary_path, 'rb') as f:
        IF_dict = pkl.load(f)


# Start the training
# =============================================================================

with tf.Graph().as_default():
    with tf.compat.v1.Session() as sess:
        classifier = cnn_clf(FLAGS)

        # Train procedure
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # Learning rate decay
        learning_rate = tf.compat.v1.train.exponential_decay(FLAGS.learning_rate, global_step, FLAGS.decay_steps,
                                                             FLAGS.decay_rate, staircase=True)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(classifier.cost)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Summaries
        loss_summary = tf.compat.v1.summary.scalar('Loss', classifier.cost)
        accuracy_summary = tf.compat.v1.summary.scalar('Accuracy', classifier.accuracy)

        # Train summary
        train_summary_op = tf.compat.v1.summary.merge_all()
        train_summary_dir = os.path.join(output_directory, 'summaries', 'train')
        train_summary_writer = tf.compat.v1.summary.FileWriter(train_summary_dir, sess.graph)

        # Validation summary
        valid_summary_op = tf.compat.v1.summary.merge_all()
        valid_summary_dir = os.path.join(output_directory, 'summaries', 'valid')
        valid_summary_writer = tf.compat.v1.summary.FileWriter(valid_summary_dir, sess.graph)

        saver = tf.compat.v1.train.Saver(max_to_keep=FLAGS.num_checkpoint)

        # initialize all the tensor variables above and from the classier class
        sess.run(tf.compat.v1.global_variables_initializer())

        # visualize the graphs with TensorBoard
        writer = tf.compat.v1.summary.FileWriter(os.path.join(output_directory, 'model', 'TensorBoard'))
        writer.add_graph(sess.graph)
        # python C:\Users\moham\Anaconda3\envs\tensorflowgpu_env\Lib\site-packages\tensorboard --logdir path_to_model_directory

        # keep results for plotting
        train_loss_results = []
        train_accuracy_results = []
        validation_loss_results = []
        validation_accuracy_results = []

        jaccard_train_loss_results = []
        jaccard_train_accuracy_results = []
        jaccard_validation_loss_results = []
        jaccard_validation_accuracy_results = []


        def run_step(input_data, is_training=True):
            # run one step of the training process
            input_x, input_y, sequence_length = input_data

            feed = {classifier.input_x: input_x, classifier.input_y: input_y}
            fetch = {'step': global_step, 'cost': classifier.cost, 'accuracy': classifier.accuracy, 'learning_rate': learning_rate}

            if FLAGS.selfie:
                interpretation_ground_truth = generate_ground_truth(input_x, input_y, inv_vocab_processor, IF_dict)
                feed[classifier.true_features] = interpretation_ground_truth
                fetch['jaccard_accuracy'] = classifier.jaccard_accuracy
                fetch['jaccard_loss'] = classifier.jaccard_loss

            if is_training:
                feed[classifier.keep_prob] = FLAGS.keep_prob
                fetch['train_op'] = train_op
                fetch['summaries'] = train_summary_op
            else:
                feed[classifier.keep_prob] = 1.0
                fetch['summaries'] = valid_summary_op

            vars = sess.run(fetch, feed)
            step = vars['step']
            loss = vars['cost']
            accuracy = vars['accuracy']
            jaccard_accuracy = vars['jaccard_accuracy'] if FLAGS.selfie else 0
            jaccard_loss = vars['jaccard_loss'] if FLAGS.selfie else 0
            summaries = vars['summaries']

            # write summaries to file
            if is_training:
                train_summary_writer.add_summary(summaries, step)
            else:
                valid_summary_writer.add_summary(summaries, step)

            if is_training:
                print(f'step: {step}, loss: {loss}, accuracy: {accuracy}, jaccard_loss: {jaccard_loss}, Jaccard accuracy: {jaccard_accuracy}')

            return loss, accuracy, jaccard_loss, jaccard_accuracy


        print('Start training ...')

        training_loss = 0
        training_accuracy = 0
        validation_loss = 0
        validation_accuracy = 0

        jaccard_training_loss = 0
        jaccard_training_accuracy = 0
        jaccard_validation_loss = 0
        jaccard_validation_accuracy = 0

        for train_input in train_data:
            t_loss, t_accuracy, t_jaccard_loss, t_jaccard_accuracy = run_step(train_input, is_training=True)
            current_step = tf.compat.v1.train.global_step(sess, global_step)
            training_loss += t_loss
            training_accuracy += t_accuracy
            jaccard_training_loss += t_jaccard_loss
            jaccard_training_accuracy += t_jaccard_accuracy

            # validate the model
            if current_step % FLAGS.evaluate_every_steps == 0:
                print('\nValidation')
                # generate sequence for the validation data
                validation_data = data_helper.batch_iter(x_valid, y_valid, valid_lengths, FLAGS.batch_size, 1)

                for validation_input in validation_data:
                    v_loss, v_accuracy, v_jaccard_loss, v_jaccard_accuracy = run_step(validation_input, is_training=False)
                    validation_loss += v_loss
                    validation_accuracy += v_accuracy
                    jaccard_validation_loss += v_jaccard_loss
                    jaccard_validation_accuracy += v_jaccard_accuracy

                print(f'step: {current_step}, '
                      f'loss: {validation_loss/(len(y_valid) // FLAGS.batch_size)}, '
                      f'accuracy: {validation_accuracy/(len(y_valid) // FLAGS.batch_size)}, '
                      f'Jaccard loss: {jaccard_validation_loss/(len(y_valid) // FLAGS.batch_size)}, '
                      f'Jaccard accuracy: {jaccard_validation_accuracy/(len(y_valid) // FLAGS.batch_size)}')

                print('Validation is over\n')

                # update the plotting results
                train_loss_results.append(training_loss/FLAGS.evaluate_every_steps)
                train_accuracy_results.append(training_accuracy/FLAGS.evaluate_every_steps)
                validation_loss_results.append(validation_loss/(len(y_valid) // FLAGS.batch_size))
                validation_accuracy_results.append(validation_accuracy/(len(y_valid) // FLAGS.batch_size))

                jaccard_train_loss_results.append(jaccard_training_loss/FLAGS.evaluate_every_steps)
                jaccard_train_accuracy_results.append(jaccard_training_accuracy/FLAGS.evaluate_every_steps)
                jaccard_validation_loss_results.append(jaccard_validation_loss/(len(y_valid) // FLAGS.batch_size))
                jaccard_validation_accuracy_results.append(jaccard_validation_accuracy/(len(y_valid) // FLAGS.batch_size))

                # reset the loss and accuracy
                training_loss = 0
                training_accuracy = 0
                validation_loss = 0
                validation_accuracy = 0

                jaccard_training_loss = 0
                jaccard_training_accuracy = 0
                jaccard_validation_loss = 0
                jaccard_validation_accuracy = 0

            # save the model to the local machine
            if current_step % FLAGS.save_every_steps == 0:
                save_path = saver.save(sess, os.path.join(output_directory, 'model/clf'), current_step)

                # plot the training / validation accuracy
                plot_results(train_loss_results, train_accuracy_results,
                             validation_loss_results, validation_accuracy_results,
                             jaccard_train_loss_results, jaccard_train_accuracy_results,
                             jaccard_validation_loss_results, jaccard_validation_accuracy_results,
                             FLAGS.evaluate_every_steps, current_step, output_directory)

        print(f'\nTraining is over... All the files have been saved to {output_directory}\n')
