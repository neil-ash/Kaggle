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
        - sibsp + parch
            - total amount of family -- no need to differentiate relation
            - remove sibsp, parch
        - adult man
            - 1 if male and age > 18, 0 otherwise
        - very young
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

# convert sex values to binary (1 == male, 0 == female)
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
train_df.Sex = labelencoder.fit_transform(train_df.Sex)


##############################################################################################################
# EXPLORE DATA
##############################################################################################################
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

# proportion of children who survived
np.sum((train_df[train_df.Age <= 5].Survived == 1).astype(int)) / len((train_df[train_df.Age <= 5]))

# total number of passengers in first, second, and third classes
len(train_df[train_df.Pclass == 1])
len(train_df[train_df.Pclass == 2])
len(train_df[train_df.Pclass == 3])

# number of passengers who survived by class: 1, 2, 3
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


Takeaways)
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



"""





'''
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

