'''
	An example multinomial classification on the MNIST data-set
'''
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_curve, roc_curve
import matplotlib
import matplotlib.pyplot as plt

# Quick method for making a simple ROC curve
def plot_roc_curve(fpr, tpr, label=None):
     plt.plot(fpr, tpr, linewidth=2, label=label)
     plt.plot([0,1], [0,1], 'k--') # Straight line
     plt.axis([0,1,0,1])
     plt.xlabel('False positive rate')
     plt.ylabel('True positive rate')

def main(debug=False):
	df_train = pd.read_csv('mnist_train.csv')
	df_test = pd.read_csv('mnist_test.csv')
	X_train, y_train = df_train.iloc[:,0], df_train.iloc[:,1:]
	X_test, y_test = df_test.iloc[:,0], df_test.iloc[:,1:]
	
	if debug:
		print(X_train.head(20))
		print(X_train.shape)
		print(y_train.head(20))
		print(y_train.shape)
	
	
	# Use a classifier based on stochastic gradient descent 
	sgd_clf = SGDClassifier(random_state=42) # As always, provide a random state
	sgd_clf.fit(X_train, y_train)
	
	'''
	One can use a cross_val_score for classification, but it's
	actually better to consider the cross_val_predict and determine
	the confusion matrix.
	'''
	cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy") 
	y_train_predict = cross_val_predict(sgd_clf, X_train, y_train, cv=5)

	# Now get the confusion matrix to quantify how often an incorrect prediction is made
	confusion_matrix(y_train, y_train_pred)

	'''
	Want the confusion matrix to be as close to diagonal as possible. Importand additional
	metrics to employ are then the precision (TP / (TP + FP)) and the recall (TP / (TP + FN)):
	'''
	precision = precision_score(y_train, y_train_predict)
	recall = recall_score(y_train, y_train_predict)
	
	# We want both high precision and high recall, combined into the harmonic mean (F1)
	harmonic_mean = fl_score(y_train, y_train_pred) # Note that there is a precision-recall tradeoff

	# Can now compute precision and recall for all possible classification scores:
	y_scores = cross_val_predict(sgd_clf, X_train, y_train, cv=3, method="decision_function")
	precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores) # From this one can select the threshold which gives the best precision/recall tradeoff for one's task

	# It is also possible to determine the ROC curve (true positive rate against fake positive rate):
	fpr, tpr, thresholds = roc_curve(y_train, y_scores)

	plot_roc_curve(fpr, tpr)
	plt.show()
    
if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-d", "--debug", help="", action="store_true", default=False)
	
    options = parser.parse_args()

    # Defining dictionary to be passed to the main function
    option_dict = dict( (k, v) for k, v in vars(options).iteritems() if v is not None)
    print option_dict
    main(**option_dict)
