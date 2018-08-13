"""
Testing various models for the MNIST dataset

Plan:
- read in data as np array/matrix
- normalize features?? probably not, since every feature is on the same scale
- one-hot encode labels
    - use data preprocessing template
- break train set into train and validation sets
    - 80/20 split (arbitraury)
- train/evaluate various models
    - logistic regression?? no need since linear SVM does same thing
    - k-nearest neighbors
    - SVM
        - linear
        - rbf
- select model with best performance on validation set
    - do not tune model hyperparameters
    - use classification accuracy as metric
- make predictions on test set
    - write predictions to csv

Notes)
- will need to do preprocessing steps on test set too: should do all at once?? Yes
- models take long time to train (10+ minutes: lots of training data)
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

# to evaluate each model
from sklearn.metrics import accuracy_score

########################################################################################################################
# K-NN
########################################################################################################################
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
from sklearn.svm import SVC
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# evaluate model
y_pred = clf.predict(X_val)
print('Linear SVM accuracy: %.2f' % accuracy_score(y_val, y_pred))


########################################################################################################################
# KERNEL (RBF) SVM
########################################################################################################################
# create and train rbf SVM
from sklearn.svm import SVC
clf = SVC(kernel='rbf')
clf.fit(X_train, y_train)

# evaluate model
y_pred = clf.predict(X_val)
print('Kernel SVM accuracy: %.2f' % accuracy_score(y_val, y_pred))


########################################################################################################################
# CHOOSE BEST MODEL AND MAKE PREDICTIONS ON TEST
########################################################################################################################
""" ONLY TRAIN BEST MODEL IN CONSOLE, THEN RUN THIS SECTION """
# predict on test set, will store predictions as 1-hot arrays
y_pred = clf.predict(X_test)

# translate from 1-hot arrays to digits (digit is the index of 1 in the array)
temp = []
unclassified = 0
for i in range(y_pred.shape[0]):
    if 1 in y_pred[i]:
        temp.append(np.where(y_pred[i] == 1)[0][0])
    # if digit is not classified, predict random value 0 -> 9 (only applicable for K-NN??)
    else:
        temp.append(np.random.randint(0, 10))
        unclassified += 1

# convert to np array, then save to csv file
y_pred = np.asarray(temp)
np.savetxt('temp_predictions.csv', y_pred, delimiter=',', fmt='%d')














