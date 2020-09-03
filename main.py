# In terminal... I installed:
#pip install sklearn
#pip install mglearn

# Import to Conda Environment

import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mglearn

# Load the Data Set
cancer = load_breast_cancer()

# STEP 1: Split the data
# Split our data set into testing and training
# Default train_size is 25% of the data that goes to testing and 75% goes to training
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=66)

# STEP 2: Set up any parameters that our algorithms call for
# Set the number of neighbors to use
clf = KNeighborsClassifier(n_neighbors = 3)

# STEP 3: Fit the training data to the specific algorithm you chose and the parameters
# Fit classifier using the training set
clf.fit(X_train, y_train)

# How good are our predictions?
print("Test.set predictions:", clf.predict(X_test))

# To evaluate how well our model generalized we can use the score method with the test data
print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))

fig, axes = plt.subplots(1,3, figsize=(10,3))
for n_neighbors, ax in zip([1, 3, 9]), axes:
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_train, y_train)

