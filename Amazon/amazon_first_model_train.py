"""
Model for Amazon Access dataset, Training Portion

Goal: create a model that takes in employee role info and resource code, outputs whether or not access should be
granted (1 or 0)

Features:
- RESOURCE: ID for each resource
    - *, 1-hot encode
- MGR_ID: employee ID of manager of current record
    - ?, may be useful, 1-hot encode
- ROLE_ROLLUP_1: company role grouping category ID 1
    - *, 1-hot encode
- ROLE_ROLLUP_2: company role grouping category ID 2
    - *, 1-hot encode
- ROLE_DEPTNAME:company role department description
    - *, 1-hot encode
- ROLE_TITLE: company role business title description
    - *, 1-hot encode
- ROLE_FAMILY_DESC: company role family extended description
    - *, 1-hot encode
- ROLE_FAMILY: company role family description
    - *, 1-hot encode
- ROLE_CODE: company role code unique to each role
    - *, 1-hot encode

Labels:
- * ACTION: whether or not resource was approved (1 for yes, 0 for no)

Plan:
- read thru and determine relevant features
- read in data w/ pandas
- save data as 2D np array
- separate features and labels
    - X and y
- 1-hot encode features
    - for both test and train **
- break training into training and validation
    - 80/20 split?
- determine appropriate model
    - kernel SVM?
    - K-NN?
- train model on training set
- evaluate model performance on validation set
    - adjust model if necessary
- make predictions on test set
- save predictions to csv file
- copy saved predictions in csv form to google sheets, correctly format
- submit

Notes)
- will need to split up train and test into separate .py files, too memory intensive for a single file
- end up w/
"""


########################################################################################################################
# LOAD AND PROCESS DATA
########################################################################################################################
import numpy as np
import pandas as pd

# features and labels for training and test sets, converted to 2D np arrays
train = pd.read_csv('data//train.csv')
train = np.asarray(train)

# X as 2D, y as 1D
X_train = train[:, 1:]
y_train = train[:, 0]

# number of training examples and number of features
m = X_train.shape[0]
n = X_train.shape[1]

# don't need to label encode, can directly 1-hot encode
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features=[0])
# 1-hot encode each training feature, want to be explicit
X0 = onehotencoder.fit_transform(X_train[:, 0].reshape(-1, 1)).toarray()
X1 = onehotencoder.fit_transform(X_train[:, 1].reshape(-1, 1)).toarray()
X2 = onehotencoder.fit_transform(X_train[:, 2].reshape(-1, 1)).toarray()
X3 = onehotencoder.fit_transform(X_train[:, 3].reshape(-1, 1)).toarray()
X4 = onehotencoder.fit_transform(X_train[:, 4].reshape(-1, 1)).toarray()
X5 = onehotencoder.fit_transform(X_train[:, 5].reshape(-1, 1)).toarray()
X6 = onehotencoder.fit_transform(X_train[:, 6].reshape(-1, 1)).toarray()
X7 = onehotencoder.fit_transform(X_train[:, 7].reshape(-1, 1)).toarray()
X8 = onehotencoder.fit_transform(X_train[:, 8].reshape(-1, 1)).toarray()
# combine all training features into one array
X_train = np.hstack((X0, X1, X2, X3, X4, X5, X6, X7, X8))

# split training set into train/validation sets 80/20
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                  y_train,
                                                  test_size=0.2,
                                                  shuffle=True)


########################################################################################################################
# TRAIN SVM
########################################################################################################################
# train w/ rbf kernel (don't know if linearly separable) on training data
from sklearn.svm import SVC
clf = SVC(kernel='rbf')
clf.fit(X_train, y_train)

# evaluate on validation set
from sklearn.metrics import accuracy_score
y_pred = clf.predict(X_val)
print('\n%.2f' % accuracy_score(y_val, y_pred))

# save trained model




