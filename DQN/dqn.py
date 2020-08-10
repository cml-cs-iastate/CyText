# dqn with keras
# Different test strategies and different tuplet selection strategies
# Author: Lei Qi
# Pre-requisite: TensorFlow version 1.12
#                Python 3

import os, sys
from sys import exit
from time import sleep
import numpy as np
import keras
import keras.backend as K
from keras.layers import Input
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras import regularizers
from keras.models import model_from_json
import tensorflow as tf

    
import warnings
warnings.filterwarnings("ignore")


def getContent(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    return lines

    
def getFileNames(dir_path):        
    files = getFilesPath(dir_path)
    fileNames = []    
    for file in files:        
        file = os.path.split(file)[1]
        fileNames.append(file)
    return fileNames

    
def getFilesPath(dir_path):
    filesPath = []
    for parent, dirnames, filenames in os.walk(dir_path):    
        for filename in filenames[:]:                        
            full_filename =  os.path.join(parent, filename)
            filesPath.append(full_filename)
    return filesPath    


# decide the character is ANSCII
def isAnscii(word):
    for elem in word:
        if ord(elem) >= 128:
            return True
    return False


# filter out non-ANSCII words    
def deleteUnAnscii(sentence):
    tempList = []
    for elem in sentence.split(" "):
        if not isAnscii(elem):
            tempList.append(elem)
    return " ".join(tempList)


def load_data(data_path):
    data_set = []
    label_set = []
    
    filenames = getFileNames(data_path)
    filenames = [filename for filename in filenames if filename.endswith('.txt')]
        
    for i, filename in enumerate(filenames):
        # print (filename)
        lines = getContent(os.path.join(data_path, filename))
        
        data_set += lines
        # label = int(filename[:-4].replace('topic', ''))
        # print(label)
        # label_set += [int(filename[:-4].replace('topic', '').strip())] * len(lines)
        # label_set += [filename[:-4]] * len(lines)
        label_set += [i] * len(lines)
        
    data_set = [line.replace("\n", "") for line in data_set]      
    shortestword = 3
    longestword = 100
    temp = []
    for line in data_set:
        line = deleteUnAnscii(line)
        line = " ".join([word for word in line.split() if longestword >= len(word) >= shortestword])
        temp.append(line)
    
    data_set[:] = temp[:]
    
    return data_set, label_set   


    
def data_pre(folder):
    
    train_data_path = folder + '/' + 'train_data/'
    
    train_data, train_label = load_data(train_data_path)
    
    data = [document for document in train_data]
    labels = train_label
    # labels = [tf.convert_to_tensor(label) for label in labels]
    # print(len(list(set(labels))))
    # print(list(set(labels)))
    # exit()
    from keras.preprocessing.text import one_hot
    from keras.preprocessing.text import text_to_word_sequence
    # define the document
    
    # estimate the size of the vocabulary
    words = set(text_to_word_sequence(' '.join(data)))
    vocab_size = len(words)
    print(vocab_size)
    
    # integer encode the document
    encode_data = []
    for i, document in enumerate(data):
        
        encode_data.append(one_hot(document, round(vocab_size)))
        
    # print(encode_data[:5])
    # print(train_label[:5])
    # print(len(encode_data))
    encode_label = keras.utils.to_categorical(labels,num_classes=None)
    # encode_label = labels
    # print(encode_label[-5:])
    # print(len(list(encode_label[0])))
    
    return encode_data, encode_label, vocab_size


def generate_tuplet(data, labels, tuplet_size):
    # data: original documents
    # labels: orginal labels for documents
    # tuplet_size: the number of documents in a tuplet
    
    data, labels = shuffle_data(data, labels)
    tuplets = []
    tuplet_labels = []
    

    for i in range(int(len(data) / tuplet_size)):
        start_id = i * tuplet_size
        end_id = start_id + tuplet_size

        tuplets.append(data[start_id: end_id])

        #create a list of ratios for all the classes
        ratio = [0]*len(labels[0])
        # print(ratio)
        
        for elem in list(labels[start_id: end_id]):
            index = list(elem).index(1.0)
            
            ratio[index] = ratio[index] + 1.0 / tuplet_size
        
        # print(ratio, len(ratio))
        # exit()
        # for i in range(len(ratio)):
            # ratio[i] = ratio[i] + 1.0 * list(labels[start_id: end_id]).count(i) / tuplet_size
        
        
        # tuplet_labels.append(tf.convert_to_tensor(ratio))
        tuplet_labels.append(np.array(ratio))
    
    
    return tuplets, tuplet_labels

# not use
def generate_batch(tuplets, tuplet_labels, batch_size):
    
    for i in range(int(len(tuplets) / batch_size)):
        start_id = i * batch_size
        end_id = start_id + batch_size

        X_batch_data.append(tuplets[start_id: end_id])
        y_batch_data.append(tuplet_labels[start_id: end_id])
        

    return X_batch_data, y_batch_data


def shuffle_data(data, labels):    
    
    shuffle_indices = np.random.permutation(np.arange(len(data)))
    data = data[shuffle_indices]
    labels = labels[shuffle_indices]   

    return data, labels


# user defined kl_divergence function
def kl_divergence(p, q):
            
    eps = 0.000001
    # p = np.array(p)
    # q = np.array(q)
    p = np.array(K.eval(p))
    q = np.array(K.eval(q))
        
    value = 0
     
    value = round((q * np.log(q/(p+eps)).sum(), 5))    
    return value


# using the builtin kl_divergence function    
def kl_divergence2(y_true, y_pred):
    
    # return keras.losses.kullback_leibler_divergence(y_true, y_pred)    # keras function
    return tf.keras.losses.KLD(y_true, y_pred)                           # tensorflow function

    
def JSD_Loss(p,q):
    m = 0.5 * (np.array(p) + np.array(q))
    
	# compute the JSD Loss
    return 0.5 * (kl_divergence2(p, m) + kl_divergence2(q, m))



def cal_mae(array1, array2):
	s = 0.0
	for i in range(len(array1)):
		s += abs(array1[i] - array2[i])
	return s / len(array1)  

def data_pre_test(folder):    
    
    test_data_path = folder + '/' + 'test_data/'
    
    test_data, test_label  = load_data(test_data_path)
    data = [document for document in test_data]
    labels = test_label
    
    # exit()
    from keras.preprocessing.text import one_hot
    from keras.preprocessing.text import text_to_word_sequence
    # define the document
    
    # estimate the size of the vocabulary
    words = set(text_to_word_sequence(' '.join(data)))
    vocab_size = len(words)
    print(vocab_size)
    
    # integer encode the document
    encode_data = []
    for i, document in enumerate(data):
        
        encode_data.append(one_hot(document, round(vocab_size)))
    
    encode_label = keras.utils.to_categorical(labels,num_classes=None)
    
    return encode_data, encode_label


def zipf(tuplet_size, num_classes):
    nums = []
    import random
    from datetime import datetime
    random.seed(datetime.now())
    rn = random.randint(0, 10)
    rn = rn * 0.1
    print('rn: ', rn)
    s = 0
    for j in range(1, num_classes+1):
        s += (j ** rn) ** (-1)

    for i in range(1, num_classes+1):
        value = int(round((((i**rn)*(s)) ** (-1)) * tuplet_size, 0))
        nums.append(value)

    # print(sum(nums))
    print(nums)
    
    return nums


def zipf_test(tuplet_size, num_classes, skew_factor):
	nums = []

	rn = skew_factor
	# print('rn: ', rn)
	s = 0
	for j in range(1, num_classes+1):
		s += (j ** rn) ** (-1)

	for i in range(1, num_classes+1):
		value = int(round((((i**rn)*(s)) ** (-1)) * tuplet_size, 0))
		nums.append(value)

	# print(sum(nums))
	# print(nums)

	return nums



def calculate_nums_binary_test(pos_ratio, tuplet_size):
	nums = []
	pos_num = int(pos_ratio*tuplet_size)
	nums.append(pos_num)
	nums.append(tuplet_size-pos_num)
	return nums	


def calculate_ratio(nums):
    
    ratio = []
    s = sum(nums)
    for num in nums:
        ratio.append(1.0 * num / s)
    
    return ratio



def generate_tuplet_zipf(data, labels, tuplet_size, num_classes):
    data, labels = shuffle_data(data, labels)
    tuplets = []
    tuplet_labels = []
        
    data_set = []
    label_set = []
    flags = []
    start_ids = []

    data_set = []
    for i in range(num_classes):
        data_set.append([])
        label_set.append([])
        flags.append([])
        start_ids.append(0)
                
    for i in range(len(labels)):
        index = list(labels[i]).index(1.0)
        data_set[index].append(data[i])
        label_set[index].append(labels[i])
        flags[index].append(False)

    while True:
        tuplet = []
        # print('here111')
        nums = zipf(tuplet_size, num_classes)
        # print(nums)
        for i in range(num_classes):
            # print(start_ids[i], len(data_set[i]))
            if start_ids[i] + nums[i] >= len(data_set[i]):
                break
            else:
                tuplet += data_set[i][start_ids[i] : start_ids[i] + nums[i]][:]
                start_ids[i] = start_ids[i] + nums[i]
        
        if len(tuplet) != tuplet_size:
            break
        elif len(tuplet) == tuplet_size:
            # print(tuplet)
            # print(len(tuplet))
            shuffle_indices = np.random.permutation(np.arange(len(tuplet)))
            # print(shuffle_indices)
            tuplet = np.array(tuplet)[shuffle_indices]
            tuplets.append(tuplet)
            ratio = calculate_ratio(nums)
            tuplet_labels.append(np.array(ratio))       
        
        # tuplet_labels.append(tf.convert_to_tensor(ratio))
        

    
    return tuplets, tuplet_labels
	


# generating training data for each epoch
# in order to feed to DQN
def generate_data(X_train, y_train, tuplet_size):	
	
	tuplets, tuplet_labels = generate_tuplet(X_train, y_train, tuplet_size)
	tuplet_labels = np.array(tuplet_labels)
	print('tuplet_labels.shape', tuplet_labels.shape)
	print('len(tuplets)', len(tuplets))
    
	data = []
	for i in range(tuplet_size):
		data.append([])

	for tuplet in tuplets:
				
		for index in range(tuplet_size):
			data[index].append(tuplet[index])       
		

	print('len(data), len(data[0]), len(data[0][0])', len(data), len(data[0]), len(data[0][0]))
	return data, tuplet_labels

  

# generating training data for each epoch
def generate_data_zipf(X_train, y_train, tuplet_size, num_classes):	
	
	tuplets, tuplet_labels = generate_tuplet_zipf(X_train, y_train, tuplet_size, num_classes)
	tuplet_labels = np.array(tuplet_labels)
	print('tuplet_labels.shape', tuplet_labels.shape)
	print('len(tuplets)', len(tuplets))
    
	data = []
	for i in range(tuplet_size):
		data.append([])

	for tuplet in tuplets:
				
		for index in range(tuplet_size):
			data[index].append(tuplet[index])       
		

	print('len(data), len(data[0]), len(data[0][0])', len(data), len(data[0]), len(data[0][0]))
	return data, tuplet_labels
    

# given pos_ratio, generate the test data that match the given pos_ratio
# data contain the original test documents; labels contain original document labels
# Return: tuplets and tuplet labels which are class ratios
def generate_tuplet_binary_test(data, labels, tuplet_size, num_classes, pos_ratio):
    data, labels = shuffle_data(data, labels)
    tuplets = []
    tuplet_labels = []
        
    data_set = []
    label_set = []
    flags = []
    start_ids = []

    data_set = []
    for i in range(num_classes):
        data_set.append([])
        label_set.append([])
        flags.append([])
        start_ids.append(0)
        
    # group documents of the same class together            
    for i in range(len(labels)):
        index = list(labels[i]).index(1.0)
        data_set[index].append(data[i])
        label_set[index].append(labels[i])
        flags[index].append(False)

    while True:
        tuplet = []        
        
        #nums is a list of number of documents for each class
        nums = calculate_nums_binary_test(pos_ratio, tuplet_size)
        
        for i in range(num_classes):
            # print(start_ids[i], len(data_set[i]))
            if start_ids[i] + nums[i] >= len(data_set[i]):
                break
            else:
                tuplet += data_set[i][start_ids[i] : start_ids[i] + nums[i]][:]
                start_ids[i] = start_ids[i] + nums[i]
        
        if len(tuplet) != tuplet_size:
            break
        elif len(tuplet) == tuplet_size:
            #get a new tuplet filled up with documents
            # the permutation is not necessary since we already shuffle the data above.
            shuffle_indices = np.random.permutation(np.arange(len(tuplet)))
            # print(shuffle_indices)
            tuplet = np.array(tuplet)[shuffle_indices]
            tuplets.append(tuplet)
            ratio = calculate_ratio(nums)
            tuplet_labels.append(np.array(ratio))  
    
    return tuplets, tuplet_labels


def generate_tuplet_multi_test(data, labels, tuplet_size, num_classes, skew_factor):
    data, labels = shuffle_data(data, labels)
    tuplets = []
    tuplet_labels = []
        
    data_set = []
    label_set = []
    flags = []
    start_ids = []

    data_set = []
    for i in range(num_classes):
        data_set.append([])
        label_set.append([])
        flags.append([])
        start_ids.append(0)
                
    for i in range(len(labels)):
        index = list(labels[i]).index(1.0)
        data_set[index].append(data[i])
        label_set[index].append(labels[i])
        flags[index].append(False)

    while True:
        tuplet = []        
        
        nums = zipf_test(tuplet_size, num_classes, skew_factor)
		
        for i in range(num_classes):
            # print(start_ids[i], len(data_set[i]))
            if start_ids[i] + nums[i] >= len(data_set[i]):
                break
            else:
                tuplet += data_set[i][start_ids[i] : start_ids[i] + nums[i]][:]
                start_ids[i] = start_ids[i] + nums[i]
        
        if len(tuplet) != tuplet_size:
            break
        elif len(tuplet) == tuplet_size:
            
            shuffle_indices = np.random.permutation(np.arange(len(tuplet)))
            # print(shuffle_indices)
            tuplet = np.array(tuplet)[shuffle_indices]
            tuplets.append(tuplet)
            ratio = calculate_ratio(nums)
            tuplet_labels.append(np.array(ratio))  
    
    return tuplets, tuplet_labels






def train_rand(folder, num_classes):
	
	epochs = 200
	X_train, y_train, vocab_size = data_pre(folder)
	# truncate and pad input sequences
	max_length = 200
	X_train = sequence.pad_sequences(X_train, maxlen=max_length)


	tuplet_size = 100
	
	# data, labels = generate_data(X_train, y_train, tuplet_size)

    # define the structure to store documents
	tweets = [] 
	for index in range(tuplet_size):
		tweets.append(Input(shape=(max_length,)))


	# only keep the top n words
	top_words = vocab_size


	# create the model
	embedding_vecor_length = 150
	shared_embedd = Embedding(top_words, embedding_vecor_length, input_length=max_length)

    # define the place holder for embedding
	tweets_embedd = []
	for index in range(tuplet_size):
		tweets_embedd.append(shared_embedd(tweets[index]))

	print(len(tweets_embedd), tweets_embedd[0].shape)

		
	# reuse the same layer layer
	shared_lstm = LSTM(128, dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=regularizers.l2(0.02), bias_regularizer=regularizers.l2(0.02))

	# define the feature placeholder for storing a document embedding (document features)
	features = []
	for index in range(tuplet_size):
		features.append(shared_lstm(tweets_embedd[index]))
    
	
	# for the NN Method
	features_NN = []
	shared_NN = Dense(256, activation='relu')
	for index in range(tuplet_size):
		features_NN.append(shared_NN(features[index]))
    
	'''
	computes the maximum (element-wise) a list of inputs.
    It takes as input a list of tensors,
    all of the same shape, and returns
    a single tensor (also of the same shape).
	
	'''
    # the output length is the number of LSTM neurons
	# merged_vector = keras.layers.maximum(features)  # average ...
	merged_vector = keras.layers.concatenate(features, axis=-1) 

	

	features = Dense(256, activation='relu')(merged_vector)
	features = Dropout(0.5)(features)
	

	predictions = Dense(num_classes, activation='softmax')(features)


	# define a trainable model linking inputs to the predictions
	model = Model(inputs=tweets, outputs=predictions)
	opt = keras.optimizers.Adam(lr =0.00001, beta_1=0.9, beta_2=0.999, amsgrad=True)
	
	model.compile(loss=JSD_Loss, optimizer=opt)


	print(model.summary())       


	for epoch in range(epochs):
		print('epoch ', epoch)
		data, labels = generate_data(X_train, y_train, tuplet_size)
		model.fit(data, labels, epochs=1, batch_size=8, verbose=1)

	# serialize model to JSON
	
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("model.h5")
	print("Saved model to disk")
	
	
	# load json and create model
	opt = keras.optimizers.Adam(lr =0.00001, beta_1=0.9, beta_2=0.999, amsgrad=True)
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("model.h5")
	print("Loaded model from disk")




def train_zipf(folder, num_classes):
	epochs = 200
	X_train, y_train, vocab_size = data_pre(folder)
	# truncate and pad input sequences
	max_length = 200
	X_train = sequence.pad_sequences(X_train, maxlen=max_length)
	
	tuplet_size = 100
	
	tweets = [] 
	for index in range(tuplet_size):
		tweets.append(Input(shape=(max_length,)))


	# numpy.random.seed(40)
	# only keep the top n words
	top_words = vocab_size


	# create the model
	embedding_vecor_length = 150
	shared_embedd = Embedding(top_words, embedding_vecor_length, input_length=max_length)

	tweets_embedd = []
	for index in range(tuplet_size):
		tweets_embedd.append(shared_embedd(tweets[index]))

	print(len(tweets_embedd), tweets_embedd[0].shape)

		
	# reuse the same layer layer
	shared_lstm = LSTM(128, dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=regularizers.l2(0.02), bias_regularizer=regularizers.l2(0.02))

	# concatenate the vectors    
	features = []
	for index in range(tuplet_size):
		features.append(shared_lstm(tweets_embedd[index]))

	# for the NN Method
    # features_NN = []
    # shared_NN = Dense(256, activation='relu')
    # for index in range(tuplet_size):
        # features_NN.append(shared_NN(features[index]))
        
	# merged_vector = keras.layers.average(features)
	merged_vector = keras.layers.concatenate(features, axis=-1)
	
	features = Dense(256, activation='relu')(merged_vector)
	features = Dropout(0.5)(features)
	

	predictions = Dense(num_classes, activation='softmax')(features)


	# define a trainable model linking inputs to the predictions
	model = Model(inputs=tweets, outputs=predictions)
	opt = keras.optimizers.Adam(lr =0.00001, beta_1=0.9, beta_2=0.999, amsgrad=True)
	
	model.compile(loss=JSD_Loss, optimizer=opt)
      


	for epoch in range(epochs):
		print('epoch ', epoch)
		data, labels = generate_data_zipf(X_train, y_train, tuplet_size,num_classes)
		model.fit(data, labels, epochs=1, batch_size=8, verbose=1)

	# serialize model to JSON
	
	model_json = model.to_json()
	with open("model2.json", "w") as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("model2.h5")
	print("Saved model to disk")
	
	
	# load json and create model
	opt = keras.optimizers.Adam(lr =0.00001, beta_1=0.9, beta_2=0.999, amsgrad=True)
	json_file = open('model2.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("model2.h5")
	print("Loaded model from disk")
	
	

def test_binary_tenstimes(folder):
	# load json and create model
	opt = keras.optimizers.Adam(lr =0.00001, beta_1=0.9, beta_2=0.999, amsgrad=True)
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("model.h5")
	print("Loaded model from disk")

	# evaluate loaded model on test data
	loaded_model.compile(loss=JSD_Loss, optimizer=opt)
	loaded_model.summary()
	
	
	epochs = 100
	X_test, y_test = data_pre_test(folder)
	# truncate and pad input sequences
	max_length = 200
	
	X_test = sequence.pad_sequences(X_test, maxlen=max_length)

	tuplet_size = 100
	
	
	maes = []
	for i in range(1, 10):
		pos_ratio = i * 0.1
		truth = [pos_ratio, 1-pos_ratio]
		print('truth: ', truth)
		for j in range(10):			
			
			for k in range(epochs):
				tuplets, tuplet_labels = generate_tuplet_binary_test(X_test, y_test, tuplet_size, 2, pos_ratio)
				tuplet_labels = np.array(tuplet_labels)
				print('tuplet_labels.shape', tuplet_labels.shape)
				print('len(tuplets)', len(tuplets))
				
				data = []
				for i in range(tuplet_size):
					data.append([])

                # organize the data in the tuplets to fit the input format of the model
				for tuplet in tuplets:
							
					for index in range(tuplet_size):
						data[index].append(tuplet[index])       
					

				# print('len(data), len(data[0]), len(data[0][0])', len(data), len(data[0]), len(data[0][0]))
				
				predictions = loaded_model.predict(data)
				pos = 0.0
				# p[0]: positive class ratio
				for p in predictions:
					pos += p[0]
					
			pos = pos / 1.0 / len(predictions) / epochs
		
			pred_ratio = [pos, 1-pos]
			print(pred_ratio)
			# mae = cal_mae(truth, pred_ratio)
			# maes.append(mae)
	
	# print(sum(maes)/len(maes))
				
				
		
			
def test_zipf_tenstimes(folder, num_classes):
	# load json and create model
	opt = keras.optimizers.Adam(lr =0.00001, beta_1=0.9, beta_2=0.999, amsgrad=True)
	json_file = open('model2.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("model2.h5")
	print("Loaded model from disk")

	
	
	epochs = 100
	X_test, y_test = data_pre_test(folder)
	# truncate and pad input sequences
	max_length = 200
	
	X_test = sequence.pad_sequences(X_test, maxlen=max_length)

	tuplet_size = 100	
	
	maes = []
	for i in range(0, 10):
		skew_factor = i * 0.1
		print(skew_factor)
        # nums is a list of numbers of documents for each class using Zipf formula
		nums = zipf_test(tuplet_size, num_classes, skew_factor)
		truth = calculate_ratio(nums)
		
		print('truth: ', truth)
		
		for j in range(10):			
			pred_ratio = np.array([0.0] * num_classes)
			for k in range(epochs):
				tuplets, tuplet_labels = generate_tuplet_multi_test(X_test, y_test, tuplet_size, num_classes, skew_factor)
				tuplet_labels = np.array(tuplet_labels)
				print('tuplet_labels.shape', tuplet_labels.shape)
				print('len(tuplets)', len(tuplets))
				
				data = []
				for i in range(tuplet_size):
					data.append([])

				for tuplet in tuplets:
							
					for index in range(tuplet_size):
						data[index].append(tuplet[index])       
					

				# print('len(data), len(data[0]), len(data[0][0])', len(data), len(data[0]), len(data[0][0]))
				
				predictions = loaded_model.predict(data)
				
				
				for p in predictions:
					pred_ratio += p / (epochs * len(predictions))
			
		
		
			print(pred_ratio)
			# mae = cal_mae(truth, list(pred_ratio))
			# maes.append(mae)
	
	# print(sum(maes)/len(maes))
				
	

    
if __name__ == '__main__':
	

	# folder = './data2'
	# train_rand(folder, 2)
	# test_binary_tenstimes(folder)
	
	
	folder = './data2'
	train_zipf(folder, 2)	
	test_binary_tenstimes(folder)
	
	# folder = './data3'
	# train_rand(folder, 4)
	# test_zipf_tenstimes(folder, 4)	
	
	# folder = './data3'
	# train_zipf(folder, 4)
	# test_zipf_tenstimes(folder, 4)	

	print ("\n\ndone...\n\n")

