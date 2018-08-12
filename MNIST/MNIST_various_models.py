"""
Testing various models for the MNIST dataset

Plan:
- read in data as np array/matrix
- normalize features??? probably not, since every feature is on the same scale
- one-hot encode labels
    - use data preprocessing template
- break train set into train and validation sets
    - 80/20 split (arbitraury)
- train/evaluate various models
    - logistic regression
    - k-nearest neighbors
    - SVM (linear and RBF)
- select model with best performance on validation set
    - do not tune model hyperparameters
    - use classification accuracy as metric
- make predictions on test set
    - write predictions to csv

Note) will need to do preprocessing steps on test set too: should do all at once?? Yes
"""


########################################################################################################################
# LOAD AND PROCESS DATA
########################################################################################################################
import numpy as np
import pandas as pd

# features and labels for training and test sets, converted to 2D np arrays
train = pd.read_csv('data//train.csv')
test = pd.read_csv('data//test.csv')
train = np.asarray(train)
test = np.asarray(test)

# keep X as 2D, y as 1D
X_train = train[:, 1:]
X_test = test
y_train = train[:, 0]

# one-hot encode labels
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features=[0])
y_train = onehotencoder.fit_transform(y_train.reshape(-1, 1)).toarray()

# break training set into train/validation sets 80/20
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                  y_train,
                                                  test_size=0.2,
                                                  shuffle=True)


########################################################################################################################
# K-NN
########################################################################################################################
# to evaluate each model
from sklearn.metrics import accuracy_score

# create and train K-NN model
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)

# evaluate model
y_pred = clf.predict(X_val)
print('K-NN accuracy: %.2f' % accuracy_score(y_val, y_pred))


########################################################################################################################
# LINEAR SVM
########################################################################################################################
# create and train linear SVM
# from sklearn.svm import SVC
# clf = SVC()





# create logistic regression model
# from sklearn.metrics import accuracy_score
# from sklearn.linear_model import LogisticRegression
# clf = LogisticRegression()
# clf.fit(X_train, y_train[:, 0])
# # evaluate model
# y_pred = clf.predict(X_val)
# print('Logisitc regression accuracy: %.2f' % accuracy_score(y_val, y_pred))



'''
""" Data Preprocessing Template """

""" import 3 main packages """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style; style.use('ggplot')

""" load data """
dataset = pd.read_csv('Data.csv')

""" features and labels: X should be matrix (2D), y should be array (1D) """
# iloc (index): select all rows, all columns except last // values:  as np array
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

""" take care of missing data """
# specify a value as missing if it is NaN, replace with mean, do WRT columns
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:, 1:])                             # fit() to learn
X[:, 1:] = imputer.transform(X[:, 1:])                      # transform() to apply

""" encode categorical data """
# by default, LabelEncoder does not encode as 1-hot array
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# need to use 1-hot, specify first column of X
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
# also encode labels y
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

""" train/test split """
from sklearn.model_selection import train_test_split
# keep track of order, 20% of data as test, random state to get reproducible results, defaults to shuffle=True
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

""" feature scaling """
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
# already fitted, can just transform
X_test = sc_X.transform(X_test)
'''


















