# Word2vec Project
### Author: Nikola Baci 

## Project Overview
In this project we build a word2vec model which will help us to find similar words by understand the context of each word. For this project, I follow the 
instructions of Dr. Ganesan, for more information you can visit her [website](https://kavita-ganesan.com/gensim-word2vec-tutorial-starter-code/#.YOTue24pD0q).

The program takes the news.crawl.gz compressed-file and builds a word2vec model that can predict the similarity of two words, 
or give the top n-similar words for a given word. The traning data can be downloaded [here](https://drive.google.com/file/d/1rNsalIR8ZuyE_tUYUYDhOcGHLDkhFTiz/view?usp=sharing).

How to run it:
1. Make sure the news.crawl.gz and word2vec.py are in the same directory
2. In the terminal type: python word2vec.py

This process will take some time (5-8 minutes). The program is reading the compressed file and then it is learning the featuers for each word using a 
nerual network of one hidden layer.

At the end you will have two outputs:
1. The wordvectors.txt (not a utf-8 file) which contains all word vectors
2. Display in the terminal of similarities for the given words. This can be found in the `word2vec_results.txt`
