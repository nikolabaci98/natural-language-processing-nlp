import os
import re
import math
import sys

class_one_train_dir = os.scandir(sys.argv[1])
class_two_train_dir = os.scandir(sys.argv[2])

vocab_file = open(sys.argv[3], 'r')

class_one_label = (sys.argv[1]).split('/')[-1]
class_two_label = (sys.argv[2]).split('/')[-1]

class_one_train_path = sys.argv[1] + '/'
class_two_train_path = sys.argv[2] + '/'

class_one_test_dir = os.scandir(sys.argv[4])
class_two_test_dir = os.scandir(sys.argv[5])

class_one_test_path = sys.argv[4] + '/'
class_two_test_path = sys.argv[5] + '/'

# stopwords_file = open('stopwords.txt', 'r')
# stopwords = set()
# for word in stopwords_file:
# 	stopwords.add(word)

#------------------------------------------------------------------------------------------------------------#
def get_vocabulary(vocab_file):
	vocabulary = set() #use len(vocabulary) to find the length of the set
	for word in vocab_file:
		word = word.strip('\n')
		vocabulary.add(word)
	return vocabulary

def train_pre_process(file, dictionary, vocabulary):
	for textline in file:
		textline = lowercase_words(textline)
		words = seperate_punctuation(textline)
		words = doc_set(words)
		for word in words:
			if word in vocabulary: #build dictionary only on words that exits in the given vocab
				dictionary[word] = dictionary.get(word, 0) + 1
	return dictionary

def doc_set(words):
	s = set()
	for word in words:
		if word not in s:
			s.add(word)
	return s

def lowercase_words(sentance):
	return sentance.lower()


def seperate_punctuation(sentance):
	return re.findall(r"[\w'-]+|[^\w\s]", sentance) #this gives commas, I don't want commas for the small review
	#return re.findall(r"[\w'-]+", sentance)

def test_features_to_file(test_file, output_file):
	file = open(test_file, 'r')
	file_words = list()
	for textline in file:
		textline = lowercase_words(textline)
		words = seperate_punctuation(textline)
		words = doc_set(words)
		for word in words:
			output_file.write(word + ' ')
	output_file.write('\n')

def calculate_class_probability(class_proba, dictX, den, test_file_words):
	for word in test_file_words:
		num = dictX.get(word, 0) + 1 #add one smoothing
		class_proba = class_proba + math.log((num / den), 2)
	return class_proba


def calculate_feature_log_prob(word, dictX, denominator):
	numerator = dictX.get(word, 0) + 1
	return math.log((numerator/denominator), 2)

#------------------------------------------------------------------------------------------------------------#


vocabulary = get_vocabulary(vocab_file)

classOneCount = 0
classOneDictionary = dict()

classTwoCount = 0
classTwoDictionary = dict()

train_features_output = open('train_features_file.txt', 'w')
test_featuers_output = open('test_features_file.txt', 'w')


#------------------------------------------------------------------------------------------------------------#

for file in class_one_test_dir:
	test_featuers_output.write(class_one_label + '(' + file.name + ') ')
	test_features_to_file(file, test_featuers_output)

for file in class_two_test_dir:
	test_featuers_output.write(class_two_label + '(' + file.name + ') ')
	test_features_to_file(file, test_featuers_output)

#------------------------------------------------------------------------------------------------------------#
class_one_file_count = len(list(class_one_train_dir))
class_two_file_count = len(list(class_two_train_dir))

class_one_prior_proba =  class_one_file_count/ (class_one_file_count + class_two_file_count)
class_two_prior_proba = 1 - class_one_prior_proba

class_one_prior_proba = math.log(class_one_prior_proba, 2)
class_two_prior_proba = math.log(class_two_prior_proba, 2)

train_features_output.write(class_one_label +' ' + str(class_one_prior_proba) + '\n')
train_features_output.write(class_two_label + ' ' + str(class_two_prior_proba) + '\n')

class_one_train_dir = os.scandir(sys.argv[1])
class_two_train_dir = os.scandir(sys.argv[2])

#build class one dictionary
for file in class_one_train_dir:
	classOneCount = classOneCount + 1
	f = open(class_one_train_path+file.name, 'r')
	classOneDictionary = train_pre_process(f, classOneDictionary, vocabulary)

#build class two dictionary
for file in class_two_train_dir:
	classTwoCount = classTwoCount + 1
	f = open(class_two_train_path+file.name, 'r')
	classTwoDictionary = train_pre_process(f, classTwoDictionary, vocabulary)

class_one_den = sum( classOneDictionary.values() ) + len(vocabulary)
class_two_den = sum( classTwoDictionary.values() ) + len(vocabulary)

train_features_output.write('parameter | class ' + class_one_label + ' log-probability | class ' + class_two_label + ' log-probability\n')
for word in vocabulary:
	train_features_output.write(word)
	train_features_output.write (' ')
	train_features_output.write(str( calculate_feature_log_prob(word, classOneDictionary, class_one_den) ))
	train_features_output.write (' ')
	train_features_output.write(str( calculate_feature_log_prob(word, classTwoDictionary, class_two_den) ))
	train_features_output.write ('\n')

#-----------------------------------------------------------------------------------------------------------------------#

