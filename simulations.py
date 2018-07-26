import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn import svm

SimData = make_classification(n_samples = 100, n_features = 20, n_informative = 2, n_redundant = 2, n_repeated = 2, n_classes = 2, flip_y = 0.01) # produce a matrix of features and corresponding discrete targets

X_train, X_test, y_train, y_test = train_test_split(SimData[0], SimData[1], test_size = 0.5) # randomly sample a training set while holding out 50% of the data for testing (evaluating) our classifier
clf = svm.SVC(kernel = 'linear', C = 1).fit(X_train, y_train)
score = clf.score(X_test, y_test)
print('Accuracy with training/test split: {}'.format(round(score, 2)))

clf = svm.SVC(kernel = 'linear', C = 1)
scores = cross_val_score(clf, SimData[0], SimData[1], cv = 5) # estimate the accuracy of a linear kernel SVM by splitting the data, fitting a model and computing the score 5 consecutive times (with different splits each time)
print('Accuracy with 5-fold cross-validation: {} (+/- {})'.format(round(scores.mean(),2), round(scores.std(),2)))

scoring = ['f1', 'roc_auc'] # specifying multiple metrics for evaluation,
scores = cross_validate(clf, SimData[0], SimData[1], scoring = scoring, cv = 5, return_train_score = False) # returns a dict containing training scores, fit-times and score-times in addition to the test score
print('f1 with 5-fold cross-validation: {}'.format(round(scores['test_f1'].mean(),2)))
print('ROC AUC with 5-fold cross-validation: {}'.format(round(scores['test_roc_auc'].mean(),2)))

cv = StratifiedKFold(n_splits = 5) # The folds are made by preserving the percentage of samples for each class.
scores = cross_val_score(clf, SimData[0], SimData[1], cv = cv)
print('Accuracy with stratified 5-fold cross-validation: {})'.format(round(scores.mean(),2)))

rfecv = RFECV(estimator = clf, step = 1, cv = cv, scoring = 'accuracy')
rfecv.fit(SimData[0], SimData[1])
print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
