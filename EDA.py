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
dd = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
dd.head()


# CONVERT CATEGORICAL DATA
dd_conv = pd.get_dummies(dd, columns=['Geography','Gender'], drop_first=True)
dd_conv.head()

# SCALE DATA TO A MEAN OF ZERO AND VARIANCE OF ONE
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
dd_scaled = pd.DataFrame(scaler.fit_transform(dd_conv),columns=dd_conv.columns)
dd_scaled.head()
#dd_scaled.plot.scatter(x='NumOfProducts',y='Exited')

# SEPARATE FEATURES AND LABELS
labels = dd_conv[['Exited']]
features = dd_scaled[dd_scaled.columns]
features.drop(['Exited'], axis=1, inplace=True)

features.head()
labels.head()


# CHECK BALANCE BETWEEN CLASSES
labels['Exited'].value_counts()
# Data is not balanced
# We will use SMOTE to upsample new data for class "1"

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(features, labels)

X_res = pd.DataFrame(X_res, columns=features.columns)
X_res.head()
y_res = pd.DataFrame(y_res, columns=labels.columns)
y_res.head()

# CHECK AGAIN BALANCE BETWEEN CLASSES
y_res['Exited'].value_counts()
# CLASSES ARE NOW BALANCED

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X_res)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, y_res[['Exited']]], axis = 1)
finalDf.head()
sns.scatterplot(x="principal component 1", y="principal component 2", hue="Exited", data=finalDf)

from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2).fit_transform(X_res)
X_embeddedDf = pd.DataFrame(data = X_embedded, columns = ['XX1', 'XX2'])
finalX_embeddedDf = pd.concat([X_embeddedDf, y_res[['Exited']]], axis = 1)
finalDf.head()
sns.scatterplot(x="XX1", y="XX2", hue="Exited", data=finalX_embeddedDf)

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

clf = SVC()
clf = KNeighborsClassifier()
clf = RandomForestClassifier()
clf = AdaBoostClassifier()
clf = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)

# TRAIN THE CLASSIFIER
clf.fit(X_train, Y_train)

# MAKE PREDICTIONS ON THE TESTING SET
predictions = clf.predict(X_test)
print(accuracy_score(predictions, Y_test))

# XGBOOST OUTPERFORMS ALL OTHER MODELS WITH 88.1% ACCURACY
