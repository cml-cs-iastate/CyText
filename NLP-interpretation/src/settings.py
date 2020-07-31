import os


# Model type. See Kim Yoon's Convolutional Neural Networks for Sentence Classification, Section 3
model_type = 'rand'  # {rand, non-static} rand mean start the embedding layer randomly, non-static mean start with the word2vec embedding

# Data source
dataset = 'fold_1'  # {'Amazon', 'Yelp', 'MR'}
data_source = 'data/AmazonYelpCombined/folds/fold_1.csv'
test_split = 0.01
validation_split = 0.01

# Model path
model_name = 'cnn_model.h5'
output_directory = os.path.join('Output', dataset)
model_path = os.path.join(output_directory, model_name)
vocabulary_dict = os.path.join(output_directory, 'vocabulary.pkl')
plot_model = False

if not os.path.exists(output_directory):
    os.makedirs(output_directory)


# Model Hyper-parameters
embedding_dim = 128
filter_sizes = (2, 3, 4, 5)
target_layer_name = ('conv1d', 'conv1d_1', 'conv1d_2', 'conv1d_3')
num_filters = 100
dropout = 0.5
fully_connected_dim = 100


# Training parameters
batch_size = 400
num_epochs = 1
model_name = 'base_model'
if dataset == 'Amazon' or dataset == 'Yelp':
    num_classes = 6  # 5 stars +1 for class 0
elif dataset == 'MR':
    num_classes = 2
else:
    num_classes = 6


# Prepossessing parameters
sequence_length = 100
max_words = 30000


# Word2Vec parameters (see train_word2vec)
min_word_count = 1
context = 10


# Interpretation features dicts
if not os.path.exists(os.path.join(output_directory, 'interpretation_features_dicts')):
    os.makedirs(os.path.join(output_directory, 'interpretation_features_dicts'))

post_processed_IF = os.path.join(output_directory, 'interpretation_features_dicts', 'post_processed_IF.pkl')
IF = os.path.join(output_directory, 'interpretation_features_dicts', 'IF.pkl')
count_IF = os.path.join(output_directory, 'interpretation_features_dicts', 'count_IF.pkl')
average_IF = os.path.join(output_directory, 'interpretation_features_dicts', 'average_IF.pkl')

milp_IF = os.path.join(output_directory, 'interpretation_features_dicts', 'milp_IF.pkl')
milp_vocab = os.path.join(output_directory, 'interpretation_features_dicts', 'milp_vocab.pkl')
