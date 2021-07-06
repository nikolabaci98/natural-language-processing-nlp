
import math

def preprocessing(file_name, train_unigram_unk, isTestFile):

	output_file_name = pad_and_lower_text(file_name)

	unigram = unigram_without_unk(output_file_name)
	if isTestFile:
		unigram_unk = test_unigram_with_unk(unigram, train_unigram_unk)
	else:
		unigram_unk = train_unigram_with_unk(unigram)

	list_of_bigrams, list_of_bigrams_unk = create_bigram_list(output_file_name, unigram_unk)
	bigram = create_bigram_dict(list_of_bigrams)
	bigram_unk = create_bigram_dict(list_of_bigrams_unk)

	return unigram, unigram_unk, bigram, bigram_unk

def pad_and_lower_text(file_name):
	#open input file and create the output file
	input_file = open(file_name, 'r')
	output_file_name = file_name.split('.')[0] + '_processed.txt'
	output_file = open(output_file_name, 'w')

	#pad each sentance of input file and lower case every word
	#write the result in the created output file 
	for line in input_file:
		s = '<s> ' + line.strip('\n').lower() + ' </s>\n'
		output_file.write(s)

	input_file.close()
	output_file.close()
	return output_file_name

def unigram_without_unk(filename):
	file_to_read = open(filename, 'r')
	unigram = {}
	for line in file_to_read:
		for word in line.split():
			unigram[word] = unigram.get(word, 0) + 1

	return unigram

def test_unigram_with_unk(test_unigram, train_unigram_unk):
	test_unigram_unk = {}
	unk = '<unk>'
	test_unigram_unk[unk] = 0

	for key, value in test_unigram.items():
		if train_unigram_unk.get(key) == None:
			test_unigram_unk[unk] +=  value
		else:
			test_unigram_unk[key] = value

	return test_unigram_unk

def train_unigram_with_unk(train_unigram):
	unigram_unk = {}
	unk = '<unk>'
	unigram_unk[unk] = 0

	for key, value in train_unigram.items():
		if value == 1:
			unigram_unk[unk] += 1
		else:
			unigram_unk[key] = value

	return unigram_unk

def create_bigram_list(filename, unigram_unk):
	file_to_read = open(filename, 'r')
	bigram_list = list()
	bigram_list_unk = list()
	unk = '<unk>'

	for line in file_to_read:
		words = line.split()
		for i  in range(len(words)-1):
			bigram_token = words[i] + ' ' + words[i+1]
			bigram_list.append(bigram_token)
			
			bigram_token_unk = ''
			if words[i] in unigram_unk.keys():
				bigram_token_unk = bigram_token_unk + words[i]
			else:
				bigram_token_unk = bigram_token_unk + unk

			if words[i+1] in unigram_unk.keys():
				bigram_token_unk = bigram_token_unk + ' ' + words[i+1]
			else:
				bigram_token_unk = bigram_token_unk + ' ' + unk 
			bigram_list_unk.append(bigram_token_unk)

	file_to_read.close()
	return bigram_list, bigram_list_unk

def create_bigram_dict(bigrams):
	bigram = {}
	for key in bigrams:
		bigram[key] = bigram.get(key, 0) + 1

	return bigram



def train(unigram_unk_train, bigram_unk_train):
	unigram_model = {}
	unigram_total_count = sum(unigram_unk_train.values())
	for key in unigram_unk_train.keys():
		unigram_model[key] = math.log(unigram_unk_train.get(key) / unigram_total_count, 2)

	vocab_size = len(unigram_unk_train.keys())
	bigram_model_mle = {}
	bigram_model_addone = {}
	for key in bigram_unk_train.keys():
		words = key.split()
		bigram_model_mle[key] = math.log(bigram_unk_train.get(key) / unigram_unk_train.get(words[0]), 2)
		bigram_model_addone[key] = math.log((bigram_unk_train.get(key)+1) /(unigram_unk_train.get(words[0])+vocab_size), 2)

	return unigram_model, bigram_model_mle, bigram_model_addone



def unigram_unseen_percentage(unigram_train, unigram_test):
	words_not_in_train = unigram_test.keys() - unigram_train.keys()

	total_missing_tokens = 0
	for key in words_not_in_train:
		total_missing_tokens = total_missing_tokens + unigram_test.get(key)

	wordPercent = len(words_not_in_train) / len(unigram_test)
	tokenPercent = total_missing_tokens / sum(unigram_test.values())

	return wordPercent, tokenPercent


def bigram_unseen_percentage(bigram_unk_train, bigram_unk_test):
	unseen_bigram_tokens = 0
	for key, val in bigram_unk_test.items():
		if bigram_unk_train.get(key, 0) == 0:
			unseen_bigram_tokens = unseen_bigram_tokens + val

	## using the bigrams condtructed by text pairs + unk
	## !!!! NOT using the V^2 params !!!!
	bigram_type_percent = len(bigram_unk_test.keys() - bigram_unk_train.keys()) / len(bigram_unk_test)
	bigram_token_percent = unseen_bigram_tokens / sum(bigram_unk_test.values())
	
	return bigram_type_percent, bigram_token_percent


def calculate_unigram_log_prob(unigram_unk_train, unigram_model, sentance):
	unk = '<unk>'
	words = sentance.split()
	for i in range(len(words)):
		if words[i] not in unigram_unk_train.keys():
			words[i] = unk

	# Unigram probaility 
	param = 'Unigram log-probability parameters:\n'
	calc = 'unigram_model (s) = '
	uni_sum_log_prob = 0
	col_width = max(len(word) for word in words) + 2
	words.pop(0) #pop <s>

	for word in words:
		param = param + word + '\t' + str(unigram_unk_train.get(word)) + ' \t' + str(unigram_model.get(word)) + '\n'
		uni_sum_log_prob = uni_sum_log_prob + unigram_model.get(word)

		calc = calc + str(unigram_model.get(word))
		if word != words[len(words)-1]:
			calc = calc + ' + '

	return param, calc, uni_sum_log_prob


def calculate_bigram_mle_log_prob(unigram_unk_train, bigram_unk_train, bigram_model_mle, sentance):
	words = sentance.split()
	unk = '<unk>'
	for i in range(len(words)):
		if words[i] not in unigram_unk_train.keys():
			words[i] = unk

	n = len(words)
	bigrams = list()
	for i in range(n-1):
		bigrams.append(words[i] + ' ' + words[i+1])

	mle_sum_log_prob = 0
	zeroFlag = False
	param = 'Bigram MLE log-probability parameters:\n'
	calc = 'bigram_mle_model (s) = '
	for word in bigrams:
		param = param + word + '\t' + str(bigram_unk_train.get(word, 0)) + ' \t' + str(bigram_model_mle.get(word, 0)) + '\n'
		
		if bigram_model_mle.get(word) == None:
			zeroFlag = True
			calc = calc + 'NaN'
		else:
			mle_sum_log_prob = mle_sum_log_prob + bigram_model_mle.get(word)
			calc = calc + str(bigram_model_mle.get(word))
		if word != bigrams[len(bigrams)-1]:
			calc = calc + ' + '

	if zeroFlag:
		mle_sum_log_prob = 'undefined'
	
	return param, calc, mle_sum_log_prob



def calculate_bigram_add_one_log_prob(unigram_unk_train, bigram_unk_train, bigram_add_one_model, sentance):
	words = sentance.split()
	unk = '<unk>'
	for i in range(len(words)):
		if words[i] not in unigram_unk_train.keys():
			words[i] = unk

	n = len(words)
	bigrams = list()
	for i in range(n-1):
		bigrams.append(words[i] + ' ' + words[i+1])

	add1_sum_log_prob = 0
	param = 'Bigram Add One log-probability parameters:\n'
	calc = 'bigram_add_one_model (s) = '
	for word in bigrams:
		
		if bigram_unk_train.get(word) == None:
			count = 1
			w = word.split()
			total = unigram_unk_train.get(w[0]) + len(unigram_unk_train.keys())
			prob = math.log(count / total, 2)
			add1_sum_log_prob = add1_sum_log_prob + prob
			
			param = param + word + '\t' + str(count) + ' \t' + str(prob) + '\n'
			calc = calc + str(prob)
		else:
			add1_sum_log_prob = add1_sum_log_prob + bigram_add_one_model.get(word)

			param = param + word + '\t' + str(bigram_unk_train.get(word)+1) + ' \t' + str(bigram_add_one_model.get(word)) + '\n'
			calc = calc + str(bigram_add_one_model.get(word))
		
		if word != bigrams[len(bigrams)-1]:
			calc = calc + ' + '

	return param, calc, add1_sum_log_prob


def compute_perplexity(uni_prob, mle_prob, add1_prob, sentance):
	s = sentance.split()

	if mle_prob == 'undefined':
		mle_pp = 'undefined'
	else:
		avg = mle_prob / len(s)
		mle_pp = 2 ** (-avg)
		
	avg = add1_prob / len(s)
	add1_pp = 2 ** (-avg)

	s.pop(0)
	avg = uni_prob / len(s)
	uni_pp = 2 ** (-avg)

	return uni_pp, mle_pp, add1_pp


def compute_unigram_test_perplexity(unigram_unk_test, unigram_model):
	test_log_prob = 0
	file = open('test_processed.txt')
	unk = '<unk>'

	for line in file:
		words = line.split()
		words.pop(0)
		sentance_log_prob = 0
		for word in words:
			if word in unigram_model:
				sentance_log_prob = sentance_log_prob + unigram_model[word]
			else:
				sentance_log_prob =  sentance_log_prob + unigram_model[unk]
		test_log_prob = test_log_prob + sentance_log_prob

	test_avg = test_log_prob / (sum(unigram_unk_test.values()) - unigram_unk_test.get('<s>'))
	pp = 2 ** (-test_avg)

	return pp

def compute_bigram_mle_test_perplexity(unigram_unk_test, bigram_mle_model):
	test_log_prob = 0
	zeroFlag = False
	file = open('test_processed.txt')
	for line in file:
		words = line.split()
		for i in range(len(words)-1):
			bigram = ''
			if words[i] in unigram_unk_test.keys():
				bigram = bigram + words[i] + ' '
			else:
				bigram = bigram + '<unk> '
			if words[i+1] in unigram_unk_test.keys():
				bigram = bigram + words[i+1]
			else:
				bigram = bigram + '<unk>'
			 
			if bigram in bigram_mle_model.keys():
				test_log_prob = test_log_prob + bigram_mle_model[bigram]
			else:
				zeroFlag = True

	if zeroFlag:
		return 'undefined'
	else:
		test_avg = test_log_prob / sum(unigram_unk_test.values())
		pp = 2 ** (-test_avg)
		return pp


def compute_bigram_add1_test_perplexity(unigram_unk_train, unigram_unk_test, bigram_add_one_model):
	file = open('test_processed.txt')
	test_log_prob = 0
	v = len(unigram_unk_train.keys())
	for line in file:
		words = line.split()
		for i in range(len(words)-1):
			bigram = ''
			if words[i] in unigram_unk_test.keys():
				bigram = bigram + words[i] + ' '
			else:
				bigram = bigram + '<unk> '
			if words[i+1] in unigram_unk_test.keys():
				bigram = bigram + words[i+1]
			else:
				bigram = bigram + '<unk>'
			 
			if bigram in bigram_add_one_model:
				test_log_prob = test_log_prob + bigram_add_one_model[bigram]
			else:
				bi = bigram.split()
				test_log_prob = test_log_prob + math.log(1 / (unigram_unk_train.get(bi[0])+v), 2)

	test_avg = test_log_prob / sum(unigram_unk_test.values())
	pp = 2 ** (-test_avg)
	return pp
