import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split,RandomizedSearchCV

import xgboost as xgb
from xgboost import XGBClassifier

df = pd.read_csv('Train.csv')

# print(df.shape,df.info(),df.head())
print(df.isnull().sum())

X = df.drop(['Warehouse_block','Mode_of_Shipment','Product_importance','Gender'],axis=1)
y = df[['Reached.on.Time_Y.N']]

X_dum = df[['Warehouse_block','Mode_of_Shipment','Product_importance','Gender']]
X_dum = pd.get_dummies(X_dum)

X = pd.concat([X,X_dum],axis=1)
id = X[['ID']]
X = X.drop(['ID','Reached.on.Time_Y.N'],axis=1)
print(X.info(),y)


# 나누기

X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,random_state=42)

# 정규화

from sklearn.preprocessing import MinMaxScaler

minmax = MinMaxScaler().fit(X_train)
scaled_X_train = minmax.transform(X_train)
scaled_X_test = minmax.transform(X_test)

print(pd.DataFrame(scaled_X_train).describe())

# 모델 학습

from sklearn.linear_model import LogisticRegression

model = LogisticRegression().fit(scaled_X_train,y_train)
pred_train = model.predict(scaled_X_train)
train_score = model.score(scaled_X_train,y_train)

pred_test = model.predict(scaled_X_test)
test_score = model.score(scaled_X_test,y_test)
print(train_score,test_score)

# print(pred_test)



from sklearn.metrics import confusion_matrix,classification_report

tr_matrix = confusion_matrix(y_train,pred_train)
test_matrix = confusion_matrix(y_test,pred_test)
print(tr_matrix,'\n',test_matrix)

print(y.shape)
print(classification_report(y_train,pred_train))
print(classification_report(y_test,pred_test))

'''
# 파라미터 조정

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
dist = {'C':randint(low=0.001,high=100)}
rand_search = RandomizedSearchCV(LogisticRegression(),param_distributions=dist,cv=5,n_iter=100,return_train_score=True).fit(scaled_X_train,y_train)
print(rand_search.best_params_,rand_search.best_score_)
'''

from sklearn.metrics import roc_auc_score

result_train = roc_auc_score(y_train,model.decision_function(scaled_X_train))
result_test = roc_auc_score(y_test,model.decision_function(scaled_X_test))
print(result_train,result_test)