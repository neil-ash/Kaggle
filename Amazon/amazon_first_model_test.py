"""
Model for Amazon Access dataset, Testing Portion
"""

########################################################################################################################
# LOAD AND PROCESS DATA
########################################################################################################################
import numpy as np
import pandas as pd

test = pd.read_csv('data//test.csv')
test = np.asarray(test)

X_test = test[:, 1:]

# don't need to label encode, can directly 1-hot encode
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features=[0])
# 1-hot encode each test feature, want to be explicit
X0 = onehotencoder.fit_transform(X_test[:, 0].reshape(-1, 1)).toarray()
X1 = onehotencoder.fit_transform(X_test[:, 1].reshape(-1, 1)).toarray()
X2 = onehotencoder.fit_transform(X_test[:, 2].reshape(-1, 1)).toarray()
X3 = onehotencoder.fit_transform(X_test[:, 3].reshape(-1, 1)).toarray()
X4 = onehotencoder.fit_transform(X_test[:, 4].reshape(-1, 1)).toarray()
X5 = onehotencoder.fit_transform(X_test[:, 5].reshape(-1, 1)).toarray()
X6 = onehotencoder.fit_transform(X_test[:, 6].reshape(-1, 1)).toarray()
X7 = onehotencoder.fit_transform(X_test[:, 7].reshape(-1, 1)).toarray()
X8 = onehotencoder.fit_transform(X_test[:, 8].reshape(-1, 1)).toarray()
# combine all test features into one array
X_test = np.hstack((X0, X1, X2, X3, X4, X5, X6, X7, X8))






