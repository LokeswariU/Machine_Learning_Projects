import math
import numpy as np
import collections
import copy
import random
import pandas as pd
from matplotlib import pyplot

def gradient_boosting(x, y, max_depth=1, num_trees=20, learning_rate=0.9):
	example_count = len(x)
	h_ens = list()

	unique_y = partition(y)
	log_odds = math.log(len(unique_y[1])/len(unique_y[0]))
	log_odds_probability = math.exp(log_odds)/(1 + math.exp(log_odds))

	# Initialize all the weights to log_odds_probability
	predicted_probabilities = np.ones(example_count)
	predicted_probabilities = predicted_probabilities*log_odds_probability

	for i in range(0, num_trees):

		# residuals = [y[i] - predicted_probabilities[i] for i in range(0, example_count)]
		residuals = np.ones(example_count)

		for j in range(0, example_count):
			residuals[j] = y[j] - predicted_probabilities[j]

		# Learn a tree for the residuals
		attribute_value_pairs = {}
		decision_tree = id3(x, residuals, predicted_probabilities, attribute_value_pairs, max_depth=max_depth)

		h_ens.append((learning_rate, decision_tree))

		# Predicting examples
		for j in range(0, example_count):
			predicted_probabilities[j] = predict_example_probability_boosting(x[j], log_odds, h_ens)

	return h_ens

def id3(x, y, predicted_probabilities, attribute_value_pairs=None, depth=0, max_depth=5):
	if(len(np.unique(y))==1):
		return log_odds_of_residuals(y, predicted_probabilities)
	elif(depth == max_depth):
		return log_odds_of_residuals(y, predicted_probabilities)
	else:
		dictionary = {}
		sorted_dict = find_best_pairs(x, y)
		pair = find_next_best_pair(sorted_dict, attribute_value_pairs)
		if(pair == None):
			return log_odds_of_residuals(y, predicted_probabilities)
		left_key = pair + (True, )
		right_key = pair + (False, )
		df_with_true_check = x[x[:, pair[0]] == pair[1]]
		df_with_false_check = x[x[:, pair[0]] != pair[1]]
		y_with_true_check = y[x[:, pair[0]] == pair[1]]
		y_with_false_check = y[x[:, pair[0]] != pair[1]]
		predicted_probabilities_with_true_check = predicted_probabilities[x[:, pair[0]] == pair[1]]
		predicted_probabilities_with_false_check = predicted_probabilities[x[:, pair[0]] != pair[1]]
		if pair not in attribute_value_pairs:
			attribute_value_pairs[pair] = 0
		attribute_value_pairs[pair] = attribute_value_pairs[pair] + 1
		attribute_value_pairs_right = copy.deepcopy(attribute_value_pairs)
		dictionary[left_key] = id3(df_with_true_check, y_with_true_check, predicted_probabilities_with_true_check, attribute_value_pairs, depth+1, max_depth)
		dictionary[right_key] = id3(df_with_false_check, y_with_false_check, predicted_probabilities_with_false_check, attribute_value_pairs_right, depth+1, max_depth)
		
		return dictionary

def log_odds_of_residuals(residuals, predicted_probabilities):
	sum_of_residuals = 0
	sum_of_probability_product = 0
	for j in range(0, len(residuals)):
		sum_of_residuals = sum_of_residuals + residuals[j]
		sum_of_probability_product = sum_of_probability_product + (predicted_probabilities[j]*(1-predicted_probabilities[j]))
	return sum_of_residuals/sum_of_probability_product

def predict_example_probability_boosting(x, log_odds, h_ens):
	log_odds_sum = log_odds

	for i in range(0, len(h_ens)):
		log_odds_sum = log_odds_sum + (h_ens[i][0] * predict_example(x, h_ens[i][1]))

	final_probability = math.exp(log_odds_sum)/(1 + math.exp(log_odds_sum))
	return final_probability

def predict_example_boosting(x, log_odds, h_ens):
	prob = predict_example_probability_boosting(x, log_odds, h_ens)
	if(prob>0.5):
		return 1
	else:
		return 0 

def predict_example(x, tree):
	for key in tree:
		sub_tree = tree[key]
		feature_index = key[0]
		feature_value = key[1]
		check = key[2]

		if(check == (x[feature_index]==feature_value)):
			if(type(sub_tree) is dict):
				result = predict_example(x, sub_tree)
			else:
				result = sub_tree

	return result

def find_next_best_pair(sorted_dict, attribute_value_pairs):
	for key in sorted_dict.keys():
		if(key in attribute_value_pairs):
			continue
		else:
			return key
	return None

def partition(x):
	dictionary = {}
	for i in range(0, len(x)):
		if(x[i] in dictionary):
			dictionary.get(x[i]).append(i)
		else:
			dictionary[x[i]] = [i]

	return dictionary

def entropy(y):
	uq = np.unique(y)
	count = collections.Counter(y)
	total = sum(count.values())
	ent_y = 0

	for i in count.keys():
		prob = count[i]/total
		ent_y -= math.log2(pow(prob, prob))
	return ent_y

def mutual_information(x, y):
	conditional_entropy = 0
	unique_x = partition(x)

	for j in unique_x.keys():
		split_y = []
		for x in unique_x[j]:
			split_y.append(y[x])
		conditional_entropy += len(split_y)/len(y)*entropy(split_y)
	information_gain = entropy(y)-conditional_entropy

	return information_gain

def find_best_pairs(x, y):
	information_gain_dict = {}
	for i in range(x.shape[1]):
		temp = x[:, i]
		x_unique = partition(temp)
		for j in x_unique.keys():
			information_gain_dict[(i, j)] = mutual_information(temp==j, y)

	sorted_dict = collections.OrderedDict(sorted(information_gain_dict.items(), key=lambda z: z[1], reverse=True))
	return sorted_dict

def compute_error(y_true, y_pred):
	n = len(y_true)
	error = (1/n) * sum(y_true != y_pred)
	return error

def create_confusion_matrix(y_true, y_pred):
	classes = np.unique(y_true)
	no_of_classes = len(classes)
	confusion_matrix = np.zeros((no_of_classes, no_of_classes), np.int32)
	for i in range(len(y_true)):
		confusion_matrix[y_true[i]][y_pred[i]] = confusion_matrix[y_true[i]][y_pred[i]] + 1

	predicted_axis_labels = [None]*no_of_classes
	actual_axis_labels = [None]*no_of_classes
	for i in range(no_of_classes):
		predicted_axis_labels[i] = "Predicted {0}".format(classes[i])
		actual_axis_labels[i] = "Actual {0}".format(classes[i])

	print(pd.DataFrame(
		confusion_matrix,
		columns=predicted_axis_labels,
		index=actual_axis_labels
	))


def learning_curve(train_error, test_error, parameter):
	pyplot.figure()
	if(parameter == 'learning_rate'):
		pyplot.xlabel('Learning Rate')
		pyplot.ylabel('Error %')
		pyplot.title("Learning curve with varying Learning rate")
	if(parameter == 'max_depth'):
		pyplot.xlabel('Maximum Tree Depth')
		pyplot.ylabel('Error %')
		pyplot.title("Learning curve with varying Maximum Tree Depth")
	if(parameter == 'num_trees'):
		pyplot.xlabel('Number of Trees')
		pyplot.ylabel('Error %')
		pyplot.title("Learning curve with varying Number of Trees")	
	pyplot.plot(train_error, color='green', marker='s')
	pyplot.plot(test_error, color='red', marker='o')
	pyplot.legend(["Training Error", "Testing Error"])
	pyplot.show()

if __name__ == '__main__':

	# Load the training data
	M = np.genfromtxt('data/train.csv', missing_values=0, skip_header=0, delimiter=',', dtype=int)
	ytrn = M[:, 0]
	Xtrn = M[:, 1:]

	# Load the test data
	M = np.genfromtxt('data/test.csv', missing_values=0, skip_header=0, delimiter=',', dtype=int)
	ytst = M[:, 0]
	Xtst = M[:, 1:]

	example_count_train = len(Xtrn)
	example_count_test = len(Xtst)

	print("Gradient Boosting Results for varying learning rate:")
	print("====================================================\n")

	train_error = [None]*6
	test_error = [None]*6

	learning_rates = [0.7,0.75,0.8,0.85,0.9,0.95]

	for i in range(len(learning_rates)):
		max_depth = 5
		num_trees = 20
		print("Configuration: learning_rate = {0}, num_trees = {1} and max_depth = {2} ".format(learning_rates[i],num_trees,max_depth))
		y_pred_train = np.empty_like(ytrn)
		y_pred_test = np.empty_like(ytst)

		ensemble = gradient_boosting(Xtrn, ytrn, max_depth, num_trees, learning_rates[i])

		unique_y = partition(ytrn)
		log_odds = math.log(len(unique_y[1])/len(unique_y[0]))

		# Predicting examples
		for j in range(0, example_count_train):
			y_pred_train[j] = predict_example_boosting(Xtrn[j], log_odds, ensemble)
		for j in range(0, example_count_test):
			y_pred_test[j] = predict_example_boosting(Xtst[j], log_odds, ensemble)

		print("Confusion matrix:")
		create_confusion_matrix(ytst, y_pred_test)
		train_error[i] = compute_error(ytrn, y_pred_train)
		test_error[i] = compute_error(ytst, y_pred_test)
		print("Training error is {0:4.2f}%.".format(train_error[i]*100))
		print("Testing error is {0:4.2f}%.".format(test_error[i]*100))
		print("\n")
	
	learning_curve(train_error, test_error, 'learning_rate')


	print("Gradient Boosting Results for varying Maximum Tree Depth:")
	print("=========================================================\n")

	train_error = [None]*5
	test_error = [None]*5

	max_depths = [1,2,3,4,5]

	for i in range(len(max_depths)):
		learning_rate = 0.3
		num_trees = 20
		
		print("Configuration: learning_rate = {0}, num_trees = {1} and max_depth = {2} ".format(learning_rate,num_trees,max_depths[i]))
		y_pred_train = np.empty_like(ytrn)
		y_pred_test = np.empty_like(ytst)


		ensemble = gradient_boosting(Xtrn, ytrn, max_depths[i], num_trees, learning_rate)

		unique_y = partition(ytrn)
		log_odds = math.log(len(unique_y[1])/len(unique_y[0]))

		# Predicting examples
		for j in range(0, example_count_train):
			y_pred_train[j] = predict_example_boosting(Xtrn[j], log_odds, ensemble)
		for j in range(0, example_count_test):
			y_pred_test[j] = predict_example_boosting(Xtst[j], log_odds, ensemble)
		
		print("Confusion matrix:")
		create_confusion_matrix(ytst, y_pred_test)
		train_error[i] = compute_error(ytrn, y_pred_train)
		test_error[i] = compute_error(ytst, y_pred_test)
		print("Training error is {0:4.2f}%.".format(train_error[i]*100))
		print("Testing error is {0:4.2f}%.".format(test_error[i]*100))
		print("\n")
	
	learning_curve(train_error, test_error, 'max_depth')

	print("Gradient Boosting Results for varying num_trees:")
	print("================================================\n")

	train_error = [None]*5
	test_error = [None]*5

	num_trees = [15,20,25,30,35]

	for i in range(len(num_trees)):
		max_depth = 5
		learning_rate = 0.3
		
		print("Configuration: learning_rate = {0}, num_trees = {1} and max_depth = {2} ".format(learning_rate,num_trees[i],max_depth))
		y_pred_train = np.empty_like(ytrn)
		y_pred_test = np.empty_like(ytst)


		ensemble = gradient_boosting(Xtrn, ytrn, max_depth, num_trees[i], learning_rate)

		unique_y = partition(ytrn)
		log_odds = math.log(len(unique_y[1])/len(unique_y[0]))

		# Predicting examples
		for j in range(0, example_count_train):
			y_pred_train[j] = predict_example_boosting(Xtrn[j], log_odds, ensemble)
		for j in range(0, example_count_test):
			y_pred_test[j] = predict_example_boosting(Xtst[j], log_odds, ensemble)

		print("Confusion matrix:")
		create_confusion_matrix(ytst, y_pred_test)
		train_error[i] = compute_error(ytrn, y_pred_train)
		test_error[i] = compute_error(ytst, y_pred_test)
		print("Training error is {0:4.2f}%.".format(train_error[i]*100))
		print("Testing error is {0:4.2f}%.".format(test_error[i]*100))
		print("\n")

	learning_curve(train_error, test_error, 'num_trees')
