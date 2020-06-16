# Vineeth Karanam, VXK180045
# Lokeswari Umakanthan, LXU190000

import math
import numpy as np
import collections
import copy
from matplotlib import pyplot
import random
import pandas as pd

def bagging(x, y, max_depth, num_trees):
	h_ens = [None]*num_trees

	for i in range(0, num_trees):
		x_bootstrap = np.empty_like(x)
		y_bootstrap = np.empty_like(y)
		for j in range(0, len(x)):
			random_row = random.randrange(0, len(x), 1)
			x_bootstrap[j] = x[random_row]
			y_bootstrap[j] = y[random_row]

		attribute_value_pairs = {}
		decision_tree = id3(x_bootstrap, y_bootstrap, attribute_value_pairs, max_depth=max_depth)
		h_ens[i] = (1, decision_tree)

	return h_ens

def boosting(x, y, max_depth=1, num_stumps=20):
	example_count = len(x)
	h_ens = [None]*num_stumps

	# Initialize all the weights to 1/N
	weights = np.ones(example_count)
	weights = weights/example_count

	x_bootstrap = x.copy()
	y_bootstrap = y.copy()

	for i in range(0, num_stumps):

		attribute_value_pairs = {}
		decision_tree = id3(x_bootstrap, y_bootstrap, attribute_value_pairs, max_depth=max_depth)

		# Predicting examples
		y_pred = np.empty_like(y)
		for j in range(0, example_count):
			y_pred[j] = predict_example(x_bootstrap[j], decision_tree)

		# Computing the error & alpha and add the tree to the list
		error = compute_error(y, y_pred)
		alpha = 0.5*(math.log((1-error)/(error)))
		h_ens[i] = (alpha, decision_tree)

		# Change the weights & normalize them
		weights = np.ones(example_count)
		weights = weights/example_count

		# Create a cumulative_weights array to make regions
		cumulative_weights = np.zeros(example_count+1)
		sum = 0
		for j in range(1, example_count+1):
			cumulative_weights[j] = sum + weights[j-1]
			sum = sum + weights[j-1]

		# Get the uniform distribution
		uniform_distribution = np.random.uniform(0, 1, example_count)

		# Generate sample based on weights
		for j in range(0, example_count):
			# Find in which region the uniform distribution's element lies in
			index = binarySearch(cumulative_weights, 0, example_count, uniform_distribution[j])
			x_bootstrap[j] = x[index]
			y_bootstrap[j] = y[index]

		x = x_bootstrap
		y = y_bootstrap

	return h_ens

def binarySearch(arr, left, right, x):
	if right >= left:
		mid = int(left + (right-left)/2)
		if arr[mid]<=x and arr[mid+1]>x:
			return mid
		elif arr[mid] > x:
			return binarySearch(arr, left, mid-1, x)
		else: 
			return binarySearch(arr, mid+1, right, x)
	else:
		return 0

def predict_example_using_ensembles(x, h_ens):
	votes = {}
	for i in range(0, len(h_ens)):
		prediction = predict_example(x, h_ens[i][1])
		if prediction in votes:
			votes[prediction] = votes[prediction] + h_ens[i][0]
		else:
			votes[prediction] = h_ens[i][0]
	
	sorted_dict = collections.OrderedDict(sorted(votes.items(), key=lambda item: item[1]))

	for key in sorted_dict.keys():
		return key

	return 1


def partition(x):
	"""
	
	+ the column vector x into subsets indexed by its unique values (v1, ... vk)

	Returns a dictionary of the form
	{ v1: indices of x == v1,
	  v2: indices of x == v2,
	  ...
	  vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
	"""

	# INSERT YOUR CODE HERE
	dictionary = {}
	for i in range(0, len(x)):
		if(x[i] in dictionary):
			dictionary.get(x[i]).append(i)
		else:
			dictionary[x[i]] = [i]

	return dictionary

def entropy(y):
	"""
	Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

	Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
	"""

	# INSERT YOUR CODE HERE
	uq = np.unique(y)
	count = collections.Counter(y)
	total = sum(count.values())
	ent_y = 0

	for i in count.keys():
		prob = count[i]/total
		ent_y -= math.log2(pow(prob, prob))
	return ent_y

def mutual_information(x, y):
	"""
	Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
	over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
	the weighted-average entropy of EACH possible split.

	Returns the mutual information: I(x, y) = H(y) - H(y | x)
	"""

	# INSERT YOUR CODE HERE
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


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
	"""
	Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
	attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
		1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
		2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
		   value of y (majority label)
		3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
	Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
	and partitions the data set based on the values of that attribute before the next recursive call to ID3.

	The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
	to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
	(taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
	attributes with their corresponding values:
	[(x1, a),
	 (x1, b),
	 (x1, c),
	 (x2, d),
	 (x2, e)]
	 If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
	 the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

	The tree is stored as a nested dictionary, where each entry is of the form
					(attribute_index, attribute_value, True/False): subtree
	* The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
	indicates that we test if (x4 == 2) at the current node.
	* The subtree itself can be nested dictionary, or a single label (leaf node).
	* Leaf nodes are (majority) class labels

	Returns a decision tree represented as a nested dictionary, for example
	{(4, 1, False):
		{(0, 1, False):
			{(1, 1, False): 1,
			 (1, 1, True): 0},
		 (0, 1, True):
			{(1, 1, False): 0,
			 (1, 1, True): 1}},
	 (4, 1, True): 1}
	"""

	# INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
	if(len(np.unique(y))==1):
		return y[0]
	elif(depth == max_depth):
		return majority_label(y)
	else:
		dictionary = {}
		sorted_dict = find_best_pairs(x, y)
		pair = find_next_best_pair(sorted_dict, attribute_value_pairs)
		if(pair == None):
			return majority_label(y)
		left_key = pair + (True, )
		right_key = pair + (False, )
		df_with_true_check = x[x[:, pair[0]] == pair[1]]
		df_with_false_check = x[x[:, pair[0]] != pair[1]]
		y_with_true_check = y[x[:, pair[0]] == pair[1]]
		y_with_false_check = y[x[:, pair[0]] != pair[1]]
		if pair not in attribute_value_pairs:
			attribute_value_pairs[pair] = 0
		attribute_value_pairs[pair] = attribute_value_pairs[pair] + 1
		attribute_value_pairs_right = copy.deepcopy(attribute_value_pairs)
		dictionary[left_key] = id3(df_with_true_check, y_with_true_check, attribute_value_pairs, depth+1, max_depth)
		dictionary[right_key] = id3(df_with_false_check, y_with_false_check, attribute_value_pairs_right, depth+1, max_depth)
		
		return dictionary

def find_next_best_pair(sorted_dict, attribute_value_pairs):
	for key in sorted_dict.keys():
		if(key in attribute_value_pairs):
			continue
		else:
			return key
	return None

def majority_label(y):
	return collections.Counter(y).most_common()[0][0]

def predict_example(x, tree):
	"""
	Predicts the classification label for a single example x using tree by recursively descending the tree until
	a label/leaf node is reached.

	Returns the predicted label of x according to tree
	"""

	# INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
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


def compute_error(y_true, y_pred):
	"""
	Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

	Returns the error = (1/n) * sum(y_true != y_pred)
	"""

	# INSERT YOUR CODE HERE
	n = len(y_true)
	error = (1/n) * sum(y_true != y_pred)
	return error

def learning_curve(train_error, test_error, parameter):
	pyplot.figure()
	if(parameter == 'max_depth'):
		pyplot.xlabel('Maximum Tree Depth')
		pyplot.ylabel('Error %')
		pyplot.title("Learning curve with varying Maximum Tree Depth")
	if(parameter == 'num_trees'):
		pyplot.xlabel('Number of Trees')
		pyplot.ylabel('Error %')
		pyplot.title("Learning curve with varying Number of Trees")
	if(parameter == 'num_stumps'):
		pyplot.xlabel('Number of Stumps')
		pyplot.ylabel('Error %')
		pyplot.title("Learning curve with varying Number of Stumps")
	pyplot.plot(train_error, color='green', marker='s')
	pyplot.plot(test_error, color='red', marker='o')
	pyplot.legend(["Training Error", "Testing Error"])
	pyplot.show()

def visualize(tree, depth=0):
	"""
	Pretty prints (kinda ugly, but hey, it's better than nothing) the decision tree to the console. Use print(tree) to
	print the raw nested dictionary representation.
	DO NOT MODIFY THIS FUNCTION!
	"""

	if depth == 0:
		print('TREE')

	for index, split_criterion in enumerate(tree):
		sub_trees = tree[split_criterion]

		# Print the current node: split criterion
		print('|\t' * depth, end='')
		print('+-- [SPLIT: x{0} = {1}]'.format(split_criterion[0], split_criterion[1]))

		# Print the children
		if type(sub_trees) is dict:
			visualize(sub_trees, depth + 1)
		else:
			print('|\t' * (depth + 1), end='')
			print('+-- [LABEL = {0}]'.format(sub_trees))

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

	y_pred_train = np.empty_like(ytrn)
	y_pred_test = np.empty_like(ytst)

	#*******************************************************************************************#
	#**************************************** PART - A *****************************************#
	#*******************************************************************************************#

	print("Bagging Results:")
	print("================\n")

	train_error = [None]*5
	test_error = [None]*5

	num_trees = [10, 15, 20, 25, 30]

	for i in range(len(num_trees)):
		max_depth = 3

		print("Configuration: num_trees = {0} and max_depth = {1} ".format(num_trees[i],max_depth))

		ensemble = bagging(Xtrn, ytrn, max_depth, num_trees[i])

		# Predicting examples
		for j in range(0, example_count_train):
			y_pred_train[j] = predict_example_using_ensembles(Xtrn[j], ensemble)
		for j in range(0, example_count_test):
			y_pred_test[j] = predict_example_using_ensembles(Xtst[j], ensemble)

		print("Confusion matrix:")
		create_confusion_matrix(ytst, y_pred_test)
		train_error[i] = compute_error(ytrn, y_pred_train)
		test_error[i] = compute_error(ytst, y_pred_test)
		print("Training error is {0:4.2f}%.".format(train_error[i]*100))
		print("Testing error is {0:4.2f}%.".format(test_error[i]*100))
		print("\n")

	learning_curve(train_error, test_error, 'num_trees')

	max_depths = [1, 2, 3, 4]

	for i in range(len(max_depths)):
		num_trees = 10
		print("Configuration: num_trees = {0} and max_depth = {1} ".format(num_trees,max_depths[i]))

		ensemble = bagging(Xtrn, ytrn, max_depths[i], num_trees)

		# Predicting examples
		for j in range(0, example_count_train):
			y_pred_train[j] = predict_example_using_ensembles(Xtrn[j], ensemble)
		for j in range(0, example_count_test):
			y_pred_test[j] = predict_example_using_ensembles(Xtst[j], ensemble)

		print("Confusion matrix:")
		create_confusion_matrix(ytst, y_pred_test)
		train_error[i] = compute_error(ytrn, y_pred_train)
		test_error[i] = compute_error(ytst, y_pred_test)
		print("Training error is {0:4.2f}%.".format(train_error[i]*100))
		print("Testing error is {0:4.2f}%.".format(test_error[i]*100))
		print("\n")

	learning_curve(train_error, test_error, 'max_depth')

	#*******************************************************************************************#
	#**************************************** PART - B *****************************************#
	#*******************************************************************************************#

	print("ADABoost Results:")
	print("=================\n")

	train_error = [None]*5
	test_error = [None]*5

	num_stumps = [10, 15, 20, 25, 30]

	for i in range(len(num_stumps)):
		max_depth = 1

		print("Configuration: num_stumps = {0} and max_depth = {1} ".format(num_stumps[i],max_depth))

		ensemble = boosting(Xtrn, ytrn, max_depth, num_stumps[i])

		# Predicting examples
		for j in range(0, example_count_train):
			y_pred_train[j] = predict_example_using_ensembles(Xtrn[j], ensemble)
		for j in range(0, example_count_test):
			y_pred_test[j] = predict_example_using_ensembles(Xtst[j], ensemble)

		print("Confusion matrix:")
		create_confusion_matrix(ytst, y_pred_test)
		train_error[i] = compute_error(ytrn, y_pred_train)
		test_error[i] = compute_error(ytst, y_pred_test)
		print("Training error is {0:4.2f}%.".format(train_error[i]*100))
		print("Testing error is {0:4.2f}%.".format(test_error[i]*100))
		print("\n")

	learning_curve(train_error, test_error, 'num_stumps')

	# train_error = [None]*3
	# test_error = [None]*3

	# max_depths = [1, 2, 3]

	# for i in range(len(max_depths)):
	# 	num_stumps = 10
	# 	print("Configuration: num_stumps = {0} and max_depth = {1} ".format(num_stumps,max_depths[i]))

	# 	ensemble = boosting(Xtrn, ytrn, max_depths[i], num_stumps)

	# 	# Predicting examples
	# 	for j in range(0, example_count_train):
	# 		y_pred_train[j] = predict_example_using_ensembles(Xtrn[j], ensemble)
	# 	for j in range(0, example_count_test):
	# 		y_pred_test[j] = predict_example_using_ensembles(Xtst[j], ensemble)

	# 	print("Confusion matrix:")
	# 	create_confusion_matrix(ytst, y_pred_test)
	# 	train_error[i] = compute_error(ytrn, y_pred_train)
	# 	test_error[i] = compute_error(ytst, y_pred_test)
	# 	print("Training error is {0:4.2f}%.".format(train_error[i]*100))
	# 	print("Testing error is {0:4.2f}%.".format(test_error[i]*100))
	# 	print("\n")

	# learning_curve(train_error, test_error, 'max_depth')