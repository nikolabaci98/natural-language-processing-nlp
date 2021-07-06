Date: April 24th, 2021
Author: Nikola Baci

*** Naïve Bayes Classifier with Add-One Smoothing ***

This implementation is a movie review (or for that fact any other topic) classifier. The algorithm is designed in two parts, preprocessing and predicion of the algorithm.

Preprocessing is done by the prepreprocessing.py file. It takes 4 arguments in the following order:
1. the path to the training directory for the first class
2. the path to the training directory for the second class
3. the path to the vocabulary file 
4. the path to the test directory for the first class
5. the path to the test directory for the second class


In this case the directory will contain all the example files that the model will be trained and the tested on.

!!important: please name the train and test directories according to the classes you are predicting. These names will be used from the algorithm to output the label.

Prediction on the test files is done in the NB.py file. It takes 3 arguments in the following order:
1. the path to the training feature vectors file
2. the path to the test pre-processed file
3. the path to the file where the prediction will be written

The first files are found in the same directory as the preprocessing.py file and are auto-generated with the following names: train_features_file and  test_features_file

The file will print the true label of the test file, then the prediction label assigned by the NB model along with the respective log-probabilities for each class (label). At the end the accuracy is displayed to evaluate how our model performed.

.
.
.

Examples of preprocessing.py command line inputs:


/Users/nikolabaci/Desktop/NLP/Homework2/movie-review-HW2/aclImdb/train/pos
/Users/nikolabaci/Desktop/NLP/Homework2/movie-review-HW2/aclImdb/train/neg

/Users/nikolabaci/Desktop/NLP/Homework2/movie-review-HW2/aclImdb/imdb.vocab

/Users/nikolabaci/Desktop/NLP/Homework2/movie-review-HW2/aclImdb/test/pos
/Users/nikolabaci/Desktop/NLP/Homework2/movie-review-HW2/aclImdb/test/neg