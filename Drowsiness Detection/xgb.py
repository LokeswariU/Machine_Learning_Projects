import math
import numpy as np
import pandas as pd
from matplotlib import pyplot
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

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
	pyplot.plot(train_error, color='green', marker='s')
	pyplot.plot(test_error, color='red', marker='o')
	pyplot.legend(["Training Error", "Testing Error"])
	pyplot.show()

def compute_error(y_true, y_pred):
	n = len(y_true)
	error = (1/n) * sum(y_true != y_pred)
	return error

if __name__ == '__main__':

	# Load the training data
	M = np.genfromtxt('data/train.csv', missing_values=0, skip_header=0, delimiter=',', dtype=int)
	ytrn = M[:, 0]
	Xtrn = M[:, 1:]

	# Load the test data
	M = np.genfromtxt('data/test.csv', missing_values=0, skip_header=0, delimiter=',', dtype=int)
	ytst = M[:, 0]
	Xtst = M[:, 1:]

	y_pred_train = np.empty_like(ytrn)
	y_pred_test = np.empty_like(ytst)

	print("XGBoost Results for varying learning rate:")
	print("==========================================\n")

	train_error = [None]*6
	test_error = [None]*6

	learning_rates = [0.7,0.75,0.8,0.85,0.9,0.95]

	for i in range(len(learning_rates)):
		max_depth = 2
		print("Configuration: learning_rate = {0} and max_depth = {1} ".format(learning_rates[i], max_depth))

		param_dist = {'max_depth': max_depth, 'learning_rate': learning_rates[i]}
		model = XGBClassifier(**param_dist)
		model.fit(Xtrn, ytrn)

		y_pred_train = model.predict(Xtrn)
		y_pred_test = model.predict(Xtst)

		print("Confusion matrix:")
		create_confusion_matrix(ytst, y_pred_test)
		train_error[i] = compute_error(ytrn, y_pred_train)
		test_error[i] = compute_error(ytst, y_pred_test)
		print("Training error is {0:4.2f}%.".format(train_error[i]*100))
		print("Testing error is {0:4.2f}%.".format(test_error[i]*100))
		print("\n")

	learning_curve(train_error, test_error, 'learning_rate')

	print("XGBoost Results for varying Maximum Tree Depth:")
	print("===============================================\n")

	train_error = [None]*5
	test_error = [None]*5

	max_depths = [1,2,3,4,5]

	for i in range(len(max_depths)):
		learning_rate = 0.9
		print("Configuration: learning_rate = {0} and max_depth = {1} ".format(learning_rate, max_depths[i]))

		param_dist = {'max_depth': max_depths[i], 'learning_rate': learning_rate}
		model = XGBClassifier(**param_dist)
		model.fit(Xtrn, ytrn)

		y_pred_train = model.predict(Xtrn)
		y_pred_test = model.predict(Xtst)

		print("Confusion matrix:")
		create_confusion_matrix(ytst, y_pred_test)
		train_error[i] = compute_error(ytrn, y_pred_train)
		test_error[i] = compute_error(ytst, y_pred_test)
		print("Training error is {0:4.2f}%.".format(train_error[i]*100))
		print("Testing error is {0:4.2f}%.".format(test_error[i]*100))
		print("\n")

	learning_curve(train_error, test_error, 'max_depth')