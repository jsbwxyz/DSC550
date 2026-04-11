# # Load libraries
# from sklearn.model_selection import cross_val_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.datasets import make_classification

# # Generate features matrix and target vector
# X, y = make_classification(n_samples = 10000,
# n_features = 3,
# n_informative = 3,
# n_redundant = 0,
# n_classes = 2,
# random_state = 1)

# # Create logistic regression
# logit = LogisticRegression()

# # Cross-validate model using accuracy
# scores = (cross_val_score(logit, X, y, scoring="precision"))
# print(scores)

# Load libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
# Load data
digits = load_digits()

# Create feature matrix and target vector
features, target = digits.data, digits.target
# Create CV training and test scores for various training set sizes
train_sizes, train_scores, test_scores = learning_curve(# Classifier
RandomForestClassifier(),
# Feature matrix
features,
# Target vector
target,
# Number of folds
cv=10,
# Performance metric
scoring='accuracy',
# Use all computer cores
n_jobs=-1,
# Sizes of 50
# Training set
train_sizes=np.linspace(
0.009,
1.0,
50))
# Create means and standard deviations of training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
# Create means and standard deviations of test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
# Draw lines
plt.plot(train_sizes, train_mean, '--', color="#111111", label="Training score")
plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")
# Draw bands
plt.fill_between(train_sizes, train_mean - train_std,
train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std,
test_mean + test_std, color="#DDDDDD")
# Create plot
plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"),
plt.legend(loc="best")
plt.tight_layout()
plt.show()