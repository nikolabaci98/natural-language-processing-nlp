# N-gram Project
### Author: Nikola Baci

## Project Overview
In this project we will train three differenet n-gram models on the `train.txt.zip` data and we will test the model on `test.txt.zip`. The data files need
need to be unzipped. Each file is a collection of texts, one sentence per line. `train.txt.zip` contains 10,000 sentences from the NewsCrawl corpus.

The code is seperated into two files:
1. preprocessing.py
2. model_training.py

The `model_training.py` is the file that calles the methods of `preprocessing.py` and generates the file `output.txt`. 
## Terminology
- word: representation of the same words, e.g. word `love` can appear many times, but it represents only one word
- token: is each word in the given text, e.g. word `love` can appear 10 times, it means we have 10 tokens of word `love`
- log probability: it is the usual probability, but we take log base 2 to overcome underflows (very small probabilities). This will cause the probabilities
to become negative and greater than 1 (see question 5 below)
- perplexity: a special variant of the usual probability used to evaluate a language model. The lower the perpexity the better our model is perfoming
- maximum likelihood model: is simple probability, the count devidied by the total
- add-one smoothing model: add on to the count (numerator) and add |V| (vocabulary size) to the denominator. This modes is used to adjust to unseen grams

## Pre-processing
Before we feed the corpora into our models, we need to pre-preprocess the text and make it uniform.
1. Pad each sentance in the training and test corpora with a start and end symbol, `<s>` and `</s>` respectively.
2. Lowercase all words in the training and test corpora (data is already tokenized for us)
3. In the training corpora replace all words that appear only once with the token `<unk>`. In the test corpora replace all data that is not in the training 
corpora with `<unk>`.

## Models
In `model_traning.py` creates 3 models:
1. A unigram with maximum likelihood probability
2. A bigram with maximum likelihood probability
3. A bigram with add-one smoothing

## Results
1. __How many word types (unique words) are there in the training corpus? Include the padding symbols and the unknown token.__  
The number of unique words in the training corpus before replacing with <unk>: 83045  
The number of unique words in the training corpus after replacing with <unk>: 41739

2. __How many word tokens are there in the training corpus?__  
The number of tokens in the training corpus is: 2568210

3. __What percentage of word tokens and word types in the test corpus did not occur
in training before the mappping the unknown words to `<unk>`?__  
The percentage of test words that did not occur in training is: 0.036028823058446756  
The percentage of test tokens that did not occur in training is: 0.01603346113628442

4. __What percentage of bigrams (bigram types and bigram tokens) in the test corpus did not occur in training after replacing singeltons with `<unk>`?__  
The percentage of bigram types that did not appear in training set is: 0.253276955602537  
The percentage of bigram tokens that did not appear in training set is: 0.21704586493318886

5. `I look forward to hearing your reply .`  
__Compute log base 2 probability of this sentace for with 3 models. List all the parameters included in the 
computation. Map words not observed in the training corpus to the `<unk>` token.__  

__Unigram log-probability parameters:__     
|  unigram  |  count  |  probability   |
|------|------|---------------------|
|i	   | 7339 	|  -8.450963962476674  |
|look	 | 613 	 | -12.032588480668235  |
|forward	|474 	|  -12.403588495460756 | 
|to	    |53048 |	-5.597321004705777  |
|hearing	|209 	|  -13.584972612278133  |
|your	  |1217 	|  -11.043218291645285  |
|reply	|  13 	 |   -17.591892026217923  |
|.	    |  87894 |	-4.868854680279238 | 
|`</s>` | 100000 |	-4.682691269922203 | 

unigram_model (s) = -8.450963962476674 + -12.032588480668235 + -12.403588495460756 + -5.597321004705777 + -13.584972612278133 + -11.043218291645285 + -17.591892026217923 + -4.868854680279238 + -4.682691269922203  

unigram_log_prob_(sentance) = -90.25609082365423  

__Bigram MLE log-probability parameters:__  
|  bigram  |  count  |  probability   |
|------|------|---------------------|
|<s> i	    |    2006 |	  -5.639534583824631  |
|i look	    |  15 	  |  -8.93447718627382  |
|look forward	|34 	 |   -4.172280422440442  |
|forward to	  |100 	|  -2.2448870591235344  |
|to hearing	 | 6 	 |   -13.110048238932082  |
|hearing your	|0 	 |   0  |
|your reply	 | 0 	 |   0  |
|reply .	   |   0 	|    0  |
|`. </s>`     | 82888 |	-0.08460143194821208  |

bigram_mle_model (s) = -5.639534583824631 + -8.93447718627382 + -4.172280422440442 + -2.2448870591235344 + -13.110048238932082 + NaN + NaN + NaN + -0.08460143194821208  

bigram_mle_log_prob_(sentance) = undefined  

__Bigram Add One log-probability parameters:__  
|  bigram  |  count  |  probability   |
|------|------|---------------------|
|<s> i	      |  2007| 	  -6.142052348726812 | 
|i look	      |16 	 |   -11.582788837823436 | 
|look forward	|35 	 |   -10.240859462550434 | 
|forward to	  |101 	 | -8.707188259410588  |
|to hearing	  |7 	   | -13.725046665121754  |
|hearing your	|1 	   | -15.35631440692812  |
|your reply	  |1 	   | -15.390572037471506 | 
|reply .	     | 1 	  |  -15.349557686620518|  
|`. </s>`	     | 82889 	|-0.6451804614204727  |

bigram_add_one_model (s) = -6.142052348726812 + -11.582788837823436 + -10.240859462550434 + -8.707188259410588 + -13.725046665121754 + -15.35631440692812 + -15.390572037471506 + -15.349557686620518 + -0.6451804614204727  

bigram_add_one_log_prob_(sentance) = -97.13956016607362

6. __Compute the perplexity of the sentence above under each of the models.__  
Perplexity of the sentance above in all three models:  
unigram_perplexity_(sentance) = 1044.3970236213092  
bigram_mle_perplexity_(sentance) = undefined  
bigram_add_one_perplexity_(sentance) = 839.83145676326  

7. __Compute the perplexity of the entire test corpus under each of the models.__  
Perplexity of the test set in all three models:  
unigram_perplexity_(test set) = 1141.6431838536487  
bigram_mle_perplexity_(test set) = undefined  
bigram_add_one_perplexity_(test set) = 1820.8651778339497  


