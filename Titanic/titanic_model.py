""""
Model for the titanic dataset

General Plan:
- determine relevant features
- normalize features
- train SVM on train set
- evaluate SVM on test set

Specific Plan:
- choose relevant features after looking at data/thinking about problem
- read in data with pandas
- store data from pandas as numpy arrays
- check for missing values
    - replace if necessary
- normalize features w/ sklearn
- separate train set into train and validation
    - choose best C value using validation
- train SVM using sklearn
- run iterations with different kernels (linear, rbf) C values, and choose the best architecture
- evaluate on test set using SVM
    - need to first read in, normalize features from test set

All Features:
- ticket class (pclass) -- yes
- sex (sex) -- yes
- age (age) -- yes
- number of siblings/spouses aboard (sibsp) -- maybe?
- number of parents/children aboard (parch) -- maybe?
- ticket number (ticket) -- no
- passenger fare (fare) -- yes
- cabin number (cabin) -- maybe?
- port of embarkation (embarked) -- maybe?

Relevant Features:
- pclass
- sex
- age
- sibsp
- parch
- fare

Labels:
- survived or died

Other Notes:
- lots of cabin numbers missing: discard feature
- some ages missing: replace with mean value
"""

##############################################################################################################
# IMPORT PACKAGES
##############################################################################################################
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


##############################################################################################################
# LOAD AND PREPROCESS TRAINING DATA
##############################################################################################################
# read in training data from csv as pandas dataframe
train_df = pd.read_csv('train.csv')

# get relevant feature columns as np arrays
pclass = train_df.Pclass.values
sex = train_df.Sex.values
age = train_df.Age.values
sibsp = train_df.SibSp.values
parch = train_df.Parch.values
fare = train_df.Fare.values

# sex: replace with 1 for male, 0 for female
sex = np.asarray([1 if i == 'male' else 0 for i in sex])

# replace missing values in age w/ mean
age_mean = np.mean([i for i in age if not np.isnan(i)])
age = np.asarray([i if not np.isnan(i) else age_mean for i in age])

# normalize non-categorical data: age, sibsp?, parch?, fare
age = normalize(age.reshape(1, -1))
sibsp = normalize(sibsp.reshape(1, -1))
parch = normalize(parch.reshape(1, -1))
fare = normalize(fare.reshape(1, -1))

# combine all features into a single array
train_features = np.vstack((pclass.ravel(), sex.ravel(), age.ravel(), sibsp.ravel(), parch.ravel(), fare.ravel())).T

# get labels as np array
train_labels = train_df.Survived.values


##############################################################################################################
# TUNE HYPERPARAMETERS
##############################################################################################################
def find_C():
    """
    Goal: find best C value using performance on validation
    - split train set into train (80%?) and validation (20%?) sets
        - randomly shuffle first
        - keep indices aligned
    - for each shuffling, test every C value
        - save accuracy on validation set and C value
    - repeat process for 100 (1000?) iterations, with a different shuffling each time
    - choose C value with highest average accuracy on validation set
    """
    # to choose best C value: test various C values
    possible_C = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000]

    # used to get best values
    best_C = None
    best_accuracy = 0

    # iterate thru C values
    for C in possible_C:

        # train/test each C value 100 times, sum up accuracies to eventually get average
        total_accuracy = 0
        for i in range(100):
            # randomly shuffle all training data into temporary train and temporary validation sets
            X_train, X_validate, y_train, y_validate = train_test_split(train_features,
                                                                        train_labels,
                                                                        test_size=0.2,
                                                                        shuffle=True)
            # train SVM w/ given train set and C value
            clf = svm.SVC(C=C)
            clf.fit(X_train, y_train)
            # 'record' total accuracy for a given C
            total_accuracy += accuracy_score(y_validate, clf.predict(X_validate))

        # average accuracy for a given C value
        average_accuracy = total_accuracy / 100
        # if best so far, record accuracy and C
        if average_accuracy > best_accuracy:
            best_accuracy = average_accuracy
            best_C = C
    return best_C


##############################################################################################################
# TRAIN MODEL
##############################################################################################################
""" TRAIN ON TRAIN AND VALIDATION?? """
# # now, have best C value (300)
# use_C = find_C()

# can train model on all of training data
clf = svm.SVC(C=300)
clf.fit(train_features, train_labels)


##############################################################################################################
# PREPROCESS TEST DATA
##############################################################################################################
""" TEST DATA """
# to evaluate on test set, need to get test data
test_df = pd.read_csv('test.csv')

# get relevant feature columns as np arrays
pclass = test_df.Pclass.values
sex = test_df.Sex.values
age = test_df.Age.values
sibsp = test_df.SibSp.values
parch = test_df.Parch.values
fare = test_df.Fare.values

# sex: replace with 1 for male, 0 for female
sex = np.asarray([1 if i == 'male' else 0 for i in sex])

# replace missing values in age w/ mean
age_mean = np.mean([i for i in age if not np.isnan(i)])
age = np.asarray([i if not np.isnan(i) else age_mean for i in age])

# replace missing values in fare w/ mean
fare_mean = np.mean([i for i in fare if not np.isnan(i)])
fare = np.asarray([i if not np.isnan(i) else fare_mean for i in fare])

# normalize non-categorical data: age, sibsp?, parch?, fare
age = normalize(age.reshape(1, -1))
sibsp = normalize(sibsp.reshape(1, -1))
parch = normalize(parch.reshape(1, -1))
fare = normalize(fare.reshape(1, -1))


##############################################################################################################
# MAKE PREDICTIONS AND WRITE TO FILE
##############################################################################################################
# combine all features into a single array
test_features = np.vstack((pclass.ravel(), sex.ravel(), age.ravel(), sibsp.ravel(), parch.ravel(), fare.ravel())).T

# predict on test features
predictions = clf.predict(test_features).astype(int)

# save predictions as csv
np.savetxt('temp_predictions.csv', predictions, delimiter=',', fmt='%d')


