from preprocessing import *

unigram_train, unigram_unk_train, bigram_train, bigram_unk_train = preprocessing('train.txt', None, False)

unigram_test, unigram_unk_test, bigram_test, bigram_unk_test = preprocessing('test.txt', unigram_unk_train, True)

unigram_model, bigram_mle_model, bigram_add_one_model = train(unigram_unk_train, bigram_unk_train)

###--------------------------------------------------------------------------------------------------###
answers = open('output.txt', 'w')
answers.write('This file includes the answers for questions 1-7:\n\n')

answers.write('Question 1:\n')
answers.write('The number of unique words in the training corpus before replacing with <unk>: ')
answers.write(str(len(unigram_train.keys())))
answers.write('\nThe number of unique words in the training corpus after replacing with <unk>: ')
answers.write(str(len(unigram_unk_train.keys())))

answers.write('\n\n')
###--------------------------------------------------------------------------------------------------###

answers.write('Question 2:\n')
answers.write('The number of tokens in the training corpus is: ')
answers.write(str(sum(unigram_train.values())))

answers.write('\n\n')
###--------------------------------------------------------------------------------------------------###

answers.write('Question 3:\n')
wp, tp = unigram_unseen_percentage(unigram_train, unigram_test)
answers.write('The percentage of test words that did not occur in training is: ')
answers.write(str(wp))
answers.write('\n')
answers.write('The percentage of test tokens that did not occur in training is: ')
answers.write(str(tp))

answers.write('\n\n')
###--------------------------------------------------------------------------------------------------###

answers.write('Question 4:\n')
bwp, btp = bigram_unseen_percentage(bigram_unk_train, bigram_unk_test)
answers.write('The percentage of bigram types that did not appear in training set is: ')
answers.write(str(bwp))
answers.write('\n')
answers.write('The percentage of bigram tokens that did not appear in training set is: ')
answers.write(str(btp))

answers.write('\n\n')
###--------------------------------------------------------------------------------------------------###

answers.write('Question 5:\n')
sentance = '<s> i look forward to hearing your reply . </s>'

param, calc, uni_prob = calculate_unigram_log_prob(unigram_unk_train, unigram_model, sentance)
answers.write(param)
answers.write('\n')
answers.write(calc)
answers.write('\n')
answers.write('\n')
answers.write('unigram_log_prob_(sentance) = ')
answers.write(str(uni_prob))
answers.write('\n\n')

param, calc, mle_prob = calculate_bigram_mle_log_prob(unigram_unk_train, bigram_unk_train, bigram_mle_model, sentance)
answers.write(param)
answers.write('\n')
answers.write(calc)
answers.write('\n')
answers.write('\n')
answers.write('bigram_mle_log_prob_(sentance) = ')
answers.write(str(mle_prob))
answers.write('\n\n')

param, calc, add1_prob = calculate_bigram_add_one_log_prob(unigram_unk_train, bigram_unk_train, bigram_add_one_model, sentance)
answers.write(param)
answers.write('\n')
answers.write(calc)
answers.write('\n')
answers.write('\n')
answers.write('bigram_add_one_log_prob_(sentance) = ')
answers.write(str(add1_prob))
answers.write('\n')

answers.write('\n\n')
###--------------------------------------------------------------------------------------------------###

answers.write('Question 6:\n')
answers.write('Perplexity of the sentance above in all three models:\n')

uni_pp, mle_pp, add1_pp = compute_perplexity(uni_prob, mle_prob, add1_prob, sentance)

answers.write('unigram_perplexity_(sentance) = ')
answers.write(str(uni_pp))
answers.write('\n')

answers.write('bigram_mle_perplexity_(sentance) = ')
answers.write(str(mle_pp))
answers.write('\n')

answers.write('bigram_add_one_perplexity_(sentance) = ')
answers.write(str(add1_pp))
answers.write('\n')

answers.write('\n\n')
###--------------------------------------------------------------------------------------------------###

answers.write('Question 7:\n')
answers.write('Perplexity of the test set in all three models:\n')

test_uni_pp = compute_unigram_test_perplexity(unigram_unk_test, unigram_model)
test_mle_pp = compute_bigram_mle_test_perplexity(unigram_unk_test, bigram_mle_model)
test_add1_pp = compute_bigram_add1_test_perplexity(unigram_unk_train, unigram_unk_test, bigram_add_one_model)


answers.write('unigram_perplexity_(test set) = ')
answers.write(str(test_uni_pp))
answers.write('\n')

answers.write('bigram_mle_perplexity_(test set) = ')
answers.write(str(test_mle_pp))
answers.write('\n')

answers.write('bigram_add_one_perplexity_(test set) = ')
answers.write(str(test_add1_pp))


	








