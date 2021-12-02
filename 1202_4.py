# 다중분류

import numpy as np
import pandas as pd
from scipy.sparse.construct import random


df = pd.read_csv('Fvote.csv',encoding='utf-8')


print(df.shape,df.info())

X = df[df.columns[:14]]
y = df[df.columns[-1]]

# print(X,y)

# exit()

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

dist = randint(low=0.001,high=100)
dist = {'C':dist}


random_search = RandomizedSearchCV(LogisticRegression(),param_distributions=dist,n_iter=100,cv=5,return_train_score=True)
random_search.fit(X_train,y_train)

best_param = random_search.best_params_
best_score = random_search.best_score_


print(best_param,best_score)

test_score = random_search.score(X_test,y_test)

print(test_score)