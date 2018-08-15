"""
Model for the Titanic dataset (second attempt)

Plan:
- do more data exploration
    - look specifically at those who died
- optimize pipeline
    - preprocess train and test features together
- include new features
    - include port of embarkation
    - feature crosses?
        - ?? sibsp + parch
            - total amount of family -- no need to differentiate relation
            - remove sibsp, parch
        - ?? adult man
            - 1 if male and age > 18, 0 otherwise
        - ?? very young
            - 1 if age <= 3, 0 otherwise
- try variety of models
    - logistic regression
        - essentially same as linear SVM
    - kernel SVM
        - already done
    - k-nearest neighbors
        - experiment with k
    - naive bayes??
        - after learning more about it
    - neural network??
        - too many parameters for dataset
        - single hidden layer?
- evaluate models on validation set
    - choose model with best performance on validation
- then train on both train and validation sets before predicting on test

All data (in column order):
- PassengerId
- Survived
- Pclass
- Name
- Sex
- Age
    - few missing values
- SibSp
- Parch
- Ticket
- Fare
- Cabin
    - many missing values
- Embarked

REMEMBER: data science is about solving problems -- computers just make things easier
"""

##############################################################################################################
# LOAD AND PREPROCESS DATA
##############################################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style; style.use('ggplot')
import seaborn as sns

# load data as df (can access specific features by name)
train_df = pd.read_csv('data//train.csv')
test_df = pd.read_csv('data//test.csv')

# fill in missing age values with mean
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
train_df.Age = imputer.fit_transform(np.asarray(train_df.Age).reshape(-1, 1))
test_df.Age = imputer.fit_transform(np.asarray(test_df.Age).reshape(-1, 1))


##############################################################################################################
# EXPLORE DATA
##############################################################################################################
'''
# see total number of passengers
train_df.PassengerId.iloc[-1]

# see the proportion of people who survived
np.count_nonzero(train_df.Survived)

# see total number of men
np.count_nonzero(train_df.Sex)

# see number of men who survived
np.count_nonzero(train_df.Sex[train_df.Survived == 1])

# see number of women who survived
np.sum((train_df.Sex[train_df.Survived == 1] == 0).astype(int))

# see what overall mean and median age are
np.mean(train_df.Age)
np.median(train_df.Age)

# see what the mean, median age for survived/died is (1 == did survive)
np.mean(train_df.Age[train_df.Survived == 1])
np.median(train_df.Age[train_df.Survived == 1])
np.mean(train_df.Age[train_df.Survived == 0])
np.median(train_df.Age[train_df.Survived == 0])

# make histogram of all ages
pd.DataFrame.hist(train_df, column='Age')

# make histograms of ages for survived/died
pd.DataFrame.hist(train_df[train_df.Survived == 1], column='Age', bins=16, color='darkgreen')
pd.DataFrame.hist(train_df[train_df.Survived == 0], column='Age', bins=16, color='darkred')

# proportion of children (age <= 5) who survived
np.sum((train_df[train_df.Age <= 5].Survived == 1).astype(int)) / len((train_df[train_df.Age <= 5]))

# total number of passengers and number who survived in first, second, and third classes
len(train_df[train_df.Pclass == 1])
len(train_df[train_df.Pclass == 2])
len(train_df[train_df.Pclass == 3])
np.sum((train_df[train_df.Pclass == 1].Survived == 1).astype(int))
np.sum((train_df[train_df.Pclass == 2].Survived == 1).astype(int))
np.sum((train_df[train_df.Pclass == 3].Survived == 1).astype(int))

# visualize number of siblings + spouses
pd.DataFrame.hist(train_df, column='SibSp', bins=8)
pd.DataFrame.hist(train_df[train_df.Survived == 1], column='SibSp', bins=8, color='darkgreen')
pd.DataFrame.hist(train_df[train_df.Survived == 0], column='SibSp', bins=8, color='darkred')

# visualize number of parents + children
pd.DataFrame.hist(train_df, column='Parch', bins=6)
pd.DataFrame.hist(train_df[train_df.Survived == 1], column='Parch', bins=6, color='darkgreen')
pd.DataFrame.hist(train_df[train_df.Survived == 0], column='Parch', bins=6, color='darkred')

# visualize fare
pd.DataFrame.hist(train_df, column='Fare', bins=100, color='skyblue')
plt.hist(train_df[train_df.Survived == 0].Fare, bins=100, color='darkred', alpha=0.5)
plt.hist(train_df[train_df.Survived == 1].Fare, bins=100, color='darkgreen', alpha=0.5)

# count passengers from each port
np.sum((train_df.Embarked == 'S').astype(int))
np.sum((train_df.Embarked == 'C').astype(int))
np.sum((train_df.Embarked == 'Q').astype(int))
np.count_nonzero(train_df[train_df.Embarked == 'S'].Survived)
np.count_nonzero(train_df[train_df.Embarked == 'C'].Survived)
np.count_nonzero(train_df[train_df.Embarked == 'Q'].Survived)

"""
Findings)
- 891 passengers, 342 survived
- 577 men
- mean age ~29/30 for both survived and died
- 109 men survived
- 233 women survived
- 44 children (below 5 yrs old)
- 31 children survived
- 216 passengers in first class
    - 136 survived
- 184 passengers in second class
    - 87 survived
- 491 passengers in third class
    - 119 survived

Main points)
General:
- 38% survived, 62% died
- 65% male, 35% female
Survival:
- 70% of children survived (age <= 5)
- 19% men survived
- 74% women survived
- 63% first class survived
- 47% second class survived
- 24% third class survived
- SibSp: weak feature
- Parch: weak feature
- Fare: decent feature
    - higher fare == more likely to survive
- Embarked: decent feature, BUT discrimination explained by other things 

Relevant Features)
- Pclass
- Sex
- Child
    - 1 if age <= 5, 0 otherwise
    - replace age
- Fare
"""
'''


##############################################################################################################
# PREPARE FOR TRAINING
##############################################################################################################
""" IMPORTANT: PCLASS IS A CATEGORICAL VARIABLE, NEED TO 1-HOT ENCODE """
# set Pclass as 2D np array
Pclass = train_df.Pclass.values.reshape(-1, 1)
Pclass_ = test_df.Pclass.values.reshape(-1, 1)
# 1-hot encode
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
Pclass = onehotencoder.fit_transform(Pclass).toarray()
Pclass_ = onehotencoder.fit_transform(Pclass_).toarray()

""" IMPORTANT: NORMALIZE FARE """
# first, replace NaN (blank) values in test
Fare = train_df.Fare.values
Fare_ = test_df.Fare.values
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
Fare_ = imputer.fit_transform(np.asarray(Fare_).reshape(-1, 1))
# now, normalize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
Fare = scaler.fit_transform(Fare.reshape(-1, 1))
Fare_ = scaler.fit_transform(Fare_.reshape(-1, 1))

""" IMPORTANT: NORMALIZE AGE """
Age = train_df.Age.values
Age_ = train_df.Age.values
Age = scaler.fit_transform(Age.reshape(-1, 1))
Age_ = scaler.fit_transform(Age_.reshape(-1, 1))

""" IMPORTANT: CONVERT SEX """
# binary values: no need to use 1-hot
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
Sex = labelencoder.fit_transform(train_df.Sex.values)
Sex_ = labelencoder.fit_transform(test_df.Sex.values)

# combine features
X_train = np.hstack((Pclass,
                     Sex.reshape(-1, 1),
                     Age.reshape(-1, 1),
                     Fare.reshape(-1, 1)))

X_test = np.hstack((Pclass_,
                    Sex_.reshape(-1, 1),
                    Age_.reshape(-1, 1),
                    Fare_.reshape(-1, 1)))

# get labels
y_train = train_df.Survived.values

# train/validation split 80/20
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                  y_train,
                                                  test_size=0.2,
                                                  shuffle=True)


##############################################################################################################
# KERNEL SVM
##############################################################################################################
from sklearn.svm import SVC
clf = SVC(kernel='rbf', C=30)
clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
y_pred = clf.predict(X_val)
print(accuracy_score(y_val, y_pred))


##############################################################################################################
# K-NN
##############################################################################################################
# from sklearn.neighbors import KNeighborsClassifier
# clf = KNeighborsClassifier(n_neighbors=5)
# clf.fit(X_train, y_train)
#
# y_pred = clf.predict(X_val)
# print(accuracy_score(y_val, y_pred))


###
'''
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

C_ls = [1, 25, 75, 100]

for C in C_ls:

    X_temp = X_train
    y_temp = y_train
    acc_total = 0

    for i in range(1000):
        X_train, X_val, y_train, y_val = train_test_split(X_temp,
                                                          y_temp,
                                                          test_size=0.2,
                                                          shuffle=True)
        clf = SVC(kernel='rbf', C=C)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        acc_total += accuracy_score(y_val, y_pred)

    print(C, ':', acc_total / 1000)
'''
###

##############################################################################################################
# PREDICT
##############################################################################################################
# train on all data (train + validation)
X_train = np.vstack((X_train, X_val))
y_train = np.concatenate((y_train, y_val))

# train
clf = SVC(kernel='rbf', C=30)
clf.fit(X_train, y_train)

# predict on test set
y_pred = clf.predict(X_test)

np.savetxt('temp_predictions2.csv', y_pred, delimiter=',', fmt='%d')


