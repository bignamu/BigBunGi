import numpy as np
import pandas as pd
import os

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Train.csv')

# print(df.shape,df.info(),df.describe(),df.isnull().sum())

### one hot
X = pd.get_dummies(df)
y = X[['Reached.on.Time_Y.N']]

id = X[['ID']].copy()

X = X.drop(['ID','Reached.on.Time_Y.N'],axis=1)

# print(X.info(),y.info(),id.info())

### train test split
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,random_state=42)

### minmax

from sklearn.preprocessing import MinMaxScaler

mm = MinMaxScaler().fit(X_train,y_train)
scaled_train = mm.transform(X_train)
scaled_test = mm.transform(X_test)

### model

import xgboost as xgb
from xgboost import XGBClassifier


xgbc =XGBClassifier(eval_metric='mlogloss',use_label_encoder=False)
xgbc.fit(scaled_train,y_train)
pred_train = xgbc.predict(scaled_train)
train_score = xgbc.score(scaled_train,y_train)

pred_test = xgbc.predict(scaled_test)
test_score = xgbc.score(scaled_test,y_test)


print(train_score,test_score)


# metrics

from sklearn.metrics import roc_auc_score,classification_report

roc_auc = roc_auc_score(y_train,pred_train)
roc_auc_test = roc_auc_score(y_test,pred_test)
print(roc_auc,roc_auc_test)