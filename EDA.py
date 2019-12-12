%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score

PATH = "Churn_Modelling.csv"

# LOAD/CHECK DATA
df = pd.read_csv(PATH)
df.head()

# DROP NON SIGNIFICANT COLUMNS
dd = df.drop(['RowNumber', 'CustomerId', 'Surname'],
            axis=1)
dd.head()


# CONVERT CATEGORICAL DATA
dd_conv = pd.get_dummies(dd,
                        columns=['Geography','Gender'],
                        drop_first=True)
dd_conv.head()

# SCALE DATA TO A MEAN OF ZERO AND VARIANCE OF ONE
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
dd_scaled = pd.DataFrame(scaler.fit_transform(dd_conv),
                        columns=dd_conv.columns)
dd_scaled.head()
#dd_scaled.plot.scatter(x='NumOfProducts',y='Exited')

# SEPARATE FEATURES AND LABELS
labels = dd_conv[['Exited']]
features = dd_scaled[dd_scaled.columns]
features.drop(['Exited'],
            axis=1,
            inplace=True)

features.head()
labels.head()


# CHECK BALANCE BETWEEN CLASSES
labels['Exited'].value_counts()

# Data is not balanced
# We will use SMOTE to upsample new data for class "1"

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(features,
                                labels)

X_res = pd.DataFrame(X_res,
                    columns=features.columns)
X_res.head()
y_res = pd.DataFrame(y_res,
                    columns=labels.columns)
y_res.head()

# CHECK AGAIN BALANCE BETWEEN CLASSES
y_res['Exited'].value_counts()
# CLASSES ARE NOW BALANCED

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X_res)

principalDf = pd.DataFrame(data = principalComponents,
                           columns = ['principal component 1',
                                      'principal component 2'])
finalDf = pd.concat([principalDf,
                    y_res[['Exited']]],
                    axis = 1)
finalDf.head()
sns.scatterplot(x="principal component 1",
                y="principal component 2",
                hue="Exited",
                data=finalDf)

"""
# TSNE TAKES TOO MUCH TIME
from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2).fit_transform(X_res)
X_embeddedDf = pd.DataFrame(data = X_embedded,
                            columns = ['XX1', 'XX2'])
finalX_embeddedDf = pd.concat([X_embeddedDf, y_res[['Exited']]],
                                axis = 1)
finalDf.head()
sns.scatterplot(x="XX1",
                y="XX2",
                hue="Exited",
                data=finalX_embeddedDf)
"""

msk = np.random.rand(len(X_res)) < 0.8
X_train = X_res[msk]
X_test = X_res[~msk]
Y_train = y_res[msk]
Y_test = y_res[~msk]

# CREATE A CLASSIFIER
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb

svc = SVC(probability=True)
knn = KNeighborsClassifier()
rfc = RandomForestClassifier()
ada = AdaBoostClassifier()
xgb = xgb.XGBClassifier(n_estimators=300) #max_depth=3, n_estimators=300, learning_rate=0.05

# TRAIN THE CLASSIFIERS
svc.fit(X_train,
        Y_train)
knn.fit(X_train,
        Y_train)
rfc.fit(X_train,
        Y_train)
ada.fit(X_train,
        Y_train)
xgb.fit(X_train,
        Y_train)

# MAKE PREDICTIONS ON THE TESTING SET
svc_predictions = svc.predict(X_test)
print('svc accuracy',accuracy_score(svc_predictions, Y_test))

knn_predictions = knn.predict(X_test)
print('knn accuracy',accuracy_score(knn_predictions, Y_test))

rfc_predictions = rfc.predict(X_test)
print('rfc accuracy',accuracy_score(rfc_predictions, Y_test))

ada_predictions = ada.predict(X_test)
print('ada accuracy',accuracy_score(ada_predictions, Y_test))

xgb_predictions = xgb.predict(X_test)
print('xgb accuracy',accuracy_score(xgb_predictions, Y_test))

# XGBOOST OUTPERFORMS ALL OTHER MODELS WITH 90% ACCURACY

# CORRELATION BETWEEN BAD PREDICTIONS OF CLASSIFIERS
svc_wrong = 1*abs(svc_predictions-Y_test['Exited'].to_numpy())
knn_wrong = 1*abs(knn_predictions-Y_test['Exited'].to_numpy())
rfc_wrong = 1*abs(rfc_predictions-Y_test['Exited'].to_numpy())
ada_wrong = 1*abs(ada_predictions-Y_test['Exited'].to_numpy())
xgb_wrong = 1*abs(xgb_predictions-Y_test['Exited'].to_numpy())

Id_wrong = svc_wrong+knn_wrong+rfc_wrong+ada_wrong+xgb_wrong

Id_wrong = pd.DataFrame(Id_wrong,
                    columns=['bad'])

pcawrong = pca.transform(X_test)

principalDfwrong = pd.DataFrame(data = pcawrong,
                           columns = ['principal component 1',
                                      'principal component 2'])
finalDfwrong = pd.concat([principalDfwrong,
                    Id_wrong[['bad']]],
                    axis = 1)
finalDfwrong.head()
sns.scatterplot(x="principal component 1",
                y="principal component 2",
                hue="bad",
                data=finalDfwrong)

# POINTS WITH A VALUE OF 2 OR LESS CAN BE ELIMINATED USING A VOTING CLASSIFIER
# GIVEN THE NUMBER OF POINTS WITH A VALUE OF 2 OR LESS, A VOTING CLASSIFIER
# SEEMS TO BE A GOOD lDEA

from sklearn.ensemble import VotingClassifier
estimators=[('svc', svc), ('knn', knn), ('rfc', rfc), ('ada', ada), ('xgb', xgb)]
eclf1 = VotingClassifier(estimators=estimators,
                        voting='hard')
eclf1.fit(X_train,
        Y_train)

eclf1_predictions = eclf1.predict(X_test)
print('ensemble accuracy',accuracy_score(eclf1_predictions, Y_test))

# THE RESULTS OF THE VOTING CLASSIFIER (89%) ARE NOT BETTER THAN THE XGBOOST ALONE
# WE WILL TRY TO GET BETTER RESULTS USING A STACKING CLASSIFIER

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
stack = StackingClassifier(estimators=estimators,
                        final_estimator=LogisticRegression())
stack.fit(X_train,
        Y_train)
stack_predictions = stack.predict(X_test)
print('stacking accuracy',accuracy_score(stack_predictions, Y_test))

# USING THE STACKING CLASSIFIER WE MANAGE TO GET 92.3% ACCURACY ON THE TESTING SET
