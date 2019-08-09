import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

PATH = "Churn_Modelling.csv"

df = pd.read_csv(PATH)
df.head()


def preprocess(df):
    labels = df[['Exited']]

    # DROP IRRELEVANT FEATURES
    #features = df.drop(['RowNumber','CustomerId', 'Surname', 'Exited'], axis=1)
    features = df[['Geography', 'Gender', 'Age', 'NumOfProducts', 'IsActiveMember']]

    # CONVERT STRINGS TO NUMBERS
    features['Gender'] = features['Gender'].replace(["Female"], 0)
    features['Gender'] = features['Gender'].replace(["Male"], 1)

    for i,x in enumerate(pd.unique(features['Geography'])):
        #print x
        features['Geography'] = features['Geography'].replace([x], i)

    # SPLIT THE DATA TO TRAIN AND TEST
    msk = np.random.rand(len(df)) < 0.8
    X_train = features[msk]
    X_test = features[~msk]
    Y_train = labels[msk]
    Y_test = labels[~msk]

    return X_train, X_test, Y_train, Y_test

# LOAD TRAIN AND TEST DATA
X_train, X_test, Y_train, Y_test = preprocess(df)


# TRAIN THE CLASSIFIER
clf = KNeighborsClassifier(n_neighbors = 2)
clf.fit(X_train, Y_train)

# MAKE PREDICTIONS ON THE TESTING SET
predictions = clf.predict(X_test)
print(accuracy_score(predictions, Y_test))
