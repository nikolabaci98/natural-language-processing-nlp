# Naïve Bayes Classifier
### Author: Nikola Baci

## Project Overview
In this project we will implement a sentiment classifier by training a Naive Bayes classifer. The classifer will user Bag of Words feature with add-one smoothing to account for word in the test set that did not appear in the training set. The code is seperated into two files `preprocessing.py` which should be run first and `NB.py` which takes as input the output of `preprocessing.py` to make predictions.


## Terminology
- class: this classifier is build to classify two groups, positive and negative, true or false, action or drama. These groups are known as classes
- vocabulary: a list of all the words (or almost all) that are found in the traning set

## Pre-processing
First run `preprocessing.py`. It takes 5 inputs as parameters in the command line. The data can be downloaded from the Google Drive, [here](https://drive.google.com/file/d/15pzijS06NcKv4d0EgRj3P-JSuCwS3k8N/view?usp=sharing), and the file needs to be unzipped. The arguments should be given in this order:
1. the path to the training directory for the first class
2. the path to the training directory for the second class
3. the path to the vocabulary file 
4. the path to the test directory for the first class
5. the path to the test directory for the second class

Examples of preprocessing.py command line inputs:
```
~/Desktop/NLP/Homework2/movie-review-HW2/aclImdb/train/pos
~/Desktop/NLP/Homework2/movie-review-HW2/aclImdb/train/neg

~/Desktop/NLP/Homework2/movie-review-HW2/aclImdb/imdb.vocab

~/Desktop/NLP/Homework2/movie-review-HW2/aclImdb/test/pos
~/Desktop/NLP/Homework2/movie-review-HW2/aclImdb/test/neg
```

The `preprocessing.py` file will produce the following output:
1. `train_features_file.txt`: in this repo this file is zipped `train_features_file.txt.zip`
    - lists each word and the log probability that this word appears in each class
2. `test_features_file.txt`: in this repo this file is also zipped `test_features_file.txt.zip`
    - a single file that contains all the reviews. each review has the label and is lowercases and tokenized

## Model
Prediction on the test files is done by the `NB.py` file. It takes 3 arguments in the following order:
1. the path to the training feature vectors file
2. the path to the test pre-processed file
3. the path to the file where the prediction will be written

The first files are found in the same directory as the `preprocessing.py` file and are auto-generated with the following names: `train_features_file.txt` and  `test_features_file.txt`. The `NB.py` will output `prediction_file.txt`. The file will contain the true label of the test file, then the prediction label assigned by the Naive Bayes model along with the respective log-probabilities for each class (label). At the end of the the accuracy is displayed to evaluate how our model performed.

## Results
The model had an accuracy of 81.5%. An increase of 2% happens if we implement Binary Naive Bayes, a varient to plain, vanilla Naive Bayes. For more information, please download the [report](https://github.com/nikolabaci98/natural-language-processing-nlp/raw/main/naive_bayes_classifier/Baci%2C%20Nikola%20Report.pdf) and you can saftly skip __Part I__ of it. 
