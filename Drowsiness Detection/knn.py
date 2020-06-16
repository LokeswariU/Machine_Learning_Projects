import math
import numpy as np
import pandas as pd
from matplotlib import pyplot

def euclidean_distance(example1, example2):
	distance = 0.0
	for i in range(1, len(example1)):
		distance += (example1[i-1] - example2[i])**2
	return math.sqrt(distance)

def get_neighbors(xy_train, x_test, num_neighbors):
	distances = list()
	for train_row in xy_train:
		distance = euclidean_distance(x_test, train_row)
		distances.append((train_row, distance))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors

def predict_classification(xy_train, x_test, num_neighbors):
	neighbors = get_neighbors(xy_train, x_test, num_neighbors)
	output_values = [row[0] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction

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

def learning_curve(train_error, test_error):
	pyplot.figure()
	pyplot.xlabel('Number of Neighbors (k)')
	pyplot.ylabel('Error %')
	pyplot.title("Learning curve with varying k")
	pyplot.plot(train_error, color='green', marker='s')
	pyplot.plot(test_error, color='red', marker='o')
	pyplot.legend(["Training Error", "Testing Error"])
	pyplot.show()

if __name__ == '__main__':

	# train_dataset = np.genfromtxt('data/monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
	# y_train = train_dataset[:, 0]
	# x_train = train_dataset[:, 1:]

	# # Load the test data
	# test_dataset = np.genfromtxt('data/monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
	# y_test = test_dataset[:, 0]
	# x_test = test_dataset[:, 1:]

	train_dataset = np.genfromtxt('data/train_normal.csv', missing_values=0, skip_header=0, delimiter=',', dtype=int)
	y_train = train_dataset[:, 0]
	x_train = train_dataset[:, 1:]

	test_dataset = np.genfromtxt('data/test_normal.csv', missing_values=0, skip_header=0, delimiter=',', dtype=int)
	y_test = test_dataset[:, 0]
	x_test = test_dataset[:, 1:]

	train_error = [None]*23
	test_error = [None]*23

	print("KNN Results:")
	print("=============\n")

	for num_neighbors in range(3, 26):
		y_pred_train = [predict_classification(train_dataset, train_example, num_neighbors) for train_example in x_train]
		y_pred_test = [predict_classification(train_dataset, test_example, num_neighbors) for test_example in x_test]
		print("Confusion matrix when k = {0}".format(num_neighbors))
		create_confusion_matrix(y_test, y_pred_test)
		train_error[num_neighbors-3] = compute_error(y_train, y_pred_train) 
		test_error[num_neighbors-3] = compute_error(y_test, y_pred_test)
		print("Training error when k = {0} is {1:4.2f}%.".format(num_neighbors, train_error[num_neighbors-3]*100))
		print("Testing error when k = {0} is {1:4.2f}%.".format(num_neighbors, test_error[num_neighbors-3]*100))
		print()
	
	learning_curve(train_error, test_error)
