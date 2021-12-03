import pandas as pd
import numpy as np

df = pd.read_csv('Train.csv')
y = df[['Reached.on.Time_Y.N']]
df = df.drop(['ID','Reached.on.Time_Y.N'],axis=1)

X = pd.get_dummies(df)
# print(df.info(),df.describe(),df.isnull().sum())

# train_test_split

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,random_state=42)

# MinMaxScaler preprocessing

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(X_train,y_train)

scaled_X_train = scaler.transform(X_train)
scaled_X_test = scaler.transform(X_test)

# RandomForest

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier().fit(scaled_X_train,y_train)
pred_train = model.predict(scaled_X_train)
pred_test = model.predict(scaled_X_test)
score = model.score(scaled_X_test,y_test)

print(f'{score:.4f}')

from sklearn.metrics import roc_auc_score,classification_report,confusion_matrix

train_report = classification_report(y_train,pred_train)
test_report = classification_report(y_test,pred_test)

train_roc = roc_auc_score(y_train,pred_train)
test_roc = roc_auc_score(y_test,pred_test)

train_conf = confusion_matrix(y_train,pred_train)
test_conf = confusion_matrix(y_test,pred_test)

print(train_report,test_report)

print(train_roc,test_roc)

print(train_conf,'\n',test_conf)


# random search

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {'n_estimators':randint(low=100,high=1000),
                  'max_features':['auto','sqrt','log2']}

random_search = RandomizedSearchCV(RandomForestClassifier(),param_distributions=param_distribs,n_iter=20,cv=5)
random_search.fit(scaled_X_train,y_train)

print(random_search.best_params_,
random_search.best_score_,
random_search.score(scaled_X_test,y_test))