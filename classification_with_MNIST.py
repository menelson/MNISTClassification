from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_curve, roc_curve
import matplotlib
import matplotlib.pyplot as plt


# Let's first consider binary classifiers, together with various metrics
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
#X, y = mnist["data"], mnist["target"]
#print(X.shape()) # 70000 images, 728 features per image (28 * 28 pixels)
#print(y.shape()) # 70000 images

some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28) # Reshape to a 28 * 28 pixel image as it's currently a 1 * 728
plt.imshow(some_digit_image, cmap = mpl.cm.binary,
           interpolation="nearest")
plt.axis("off")
plt.show()

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
# Shuffle the training data
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# Let's test and train a "5-detector"
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

# Use a classifier based on stochastic gradient descent 
sgd_clf = SGDClassifier(random_state=42) # As always, provide a random state
sgd_clf.fit(X_train, y_train_5)

# One can use a cross_val_score for classification, but it's
# actually better to consider the cross_val_predict and determine
# the confusion matrix.
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy") # K-fold cross-validation, in this case with 3 folds
y_train_predict = cross_val_predict(sgd_clf, X_train, y_train, cv=5)
# Now get the confusion matrix to quantify how often an incorrect prediction is made
confusion_matrix(y_train_5, y_train_pred)
# Want the confusion matrix to be as close to diagonal as possible. Importand additional
# metrics to employ are then the precision (TP / (TP + FP)) and the recall (TP / (TP + FN)):
precision = precision_score(y_train_5, y_train_predict)
recall = recall_score(y_train_5, y_train_predict)
# We want both high precision and high recall, combined into the harmonic mean (F1)
harmonic_mean = fl_score(y_train_5, y_train_pred)
# Note that there is a precision-recall tradeoff

# Can obtain the classifier's score using decision_function(), which is what the
# classifier cuts on when performing the actual classification
#y_scores = sgd_clf.decision_function([some_digit]) # Get the score for a particular instance
# Can now compute precision and recall for all possible classification scores:
y_scores = cross_val_predict(sgd_clf, X_train, y_train, cv=3, method="decision_function")
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores) # From this one can select the threshold which gives the best precision/recall tradeoff for one's task

# It is also possible to determine the ROC curve (true positive rate against fake positive rate):
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

# Quick method for making a simple ROC curve
def plot_roc_curve(fpr, tpr, label=None):
     plt.plot(fpr, tpr, linewidth=2, label=label)
     plt.plot([0,1], [0,1], 'k--') # Straight line
     plt.axis([0,1,0,1])
     plt.xlabel('False positive rate')
     plt.ylabel('True positive rate')

plot_roc_curve(fpr, tpr)
plt.show()

# Note the very important ROC curve metric: the AUC (closer to 1, the better the classifier)

# 2. Binary classifier
# Can also employ additional classification techniques, such as one-versus-one or 
# one-versus-all.
# It's also useful to normalise the confusuon matrix by dividing through by the row sums, thus generating an
# error matrix.

