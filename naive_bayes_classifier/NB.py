
import sys


nb_train_parameters = open(sys.argv[1], 'r')

class_one_log_prob_features = dict()
class_two_log_prob_features = dict()

classlabel_classproba_str = next(nb_train_parameters).split()
class_one_label = classlabel_classproba_str[0]
class_one_prior_proba = float(classlabel_classproba_str[1])

classlabel_classproba_str = next(nb_train_parameters).split()
class_two_label = classlabel_classproba_str[0]
class_two_prior_proba = float(classlabel_classproba_str[1])



next(nb_train_parameters) #skip the column names

for line in nb_train_parameters:
	vector = line.split()
	class_one_log_prob_features[vector[0]] = float(vector[1])
	class_two_log_prob_features[vector[0]] = float(vector[2])

nb_test_parameters = open(sys.argv[2], 'r')
prediction_file = open(sys.argv[3], 'w')

header = 'true label | pred label | class ' + class_one_label + ' probability | class ' + class_two_label + ' probability\n'
prediction_file.write(header)


correct_prediction_count = 0
total_count = 0
neg_mis_pred = 0
pos_mis_pred = 0
for line in nb_test_parameters:
	c1_proba = class_one_prior_proba
	c2_proba = class_two_prior_proba

	vector = line.split()
	label = vector[0]
	vector.pop(0)
	prediction_file.write('x = ' + label + ' ')
	label = label.split('(')[0]
	for word in vector:
		c1_proba = c1_proba + class_one_log_prob_features.get(word, 0)
		c2_proba = c2_proba + class_two_log_prob_features.get(word, 0)

	if c1_proba > c2_proba:
		prediction = class_one_label
	else:
		prediction = class_two_label

	prediction_file.write('y = ' + prediction + ' ' + str(c1_proba) + ' ' + str(c2_proba) + '\n')

	if prediction == label:
		correct_prediction_count = correct_prediction_count + 1
	elif label == 'pos':
		pos_mis_pred = pos_mis_pred + 1
	else:
		neg_mis_pred = neg_mis_pred + 1
	total_count = total_count + 1


accuracy = (correct_prediction_count / total_count) * 100
prediction_file.write('accuracy = ' + str(accuracy) + '%\n')
prediction_file.write('missed positive = ' + str(pos_mis_pred) + '\n')
prediction_file.write('missed negative = ' + str(neg_mis_pred) + '\n')


