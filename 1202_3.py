# Gird Search // Random Search

import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split

df = pd.read_csv('Fvote.csv',encoding='utf-8')

print(df.shape,df.describe())
print(df.columns)

X = df[['gender_female', 'gender_male', 'region_Chungcheung', 'region_Honam',
       'region_Others', 'region_Sudo', 'region_Youngnam', 'edu', 'income',
       'age', 'score_gov', 'score_progress', 'score_intention',   'parties']]
y = df[['vote']]
# print(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

params_distribs = {'C':randint(low=0.001,high=100)}

random_search = RandomizedSearchCV(LogisticRegression(),param_distributions=params_distribs,cv=5,n_iter=100,return_train_score=True)

random_search.fit(X_train,y_train)


print(f'Best Parameter : {random_search.best_params_}')
print(f'Best Cross-Validation Score : {random_search.best_score_:.3f}')
print(f'Test set Score : {random_search.score(X_test,y_test):.3f}')


result_random = random_search.cv_results_
print(pd.DataFrame(result_random))

import matplotlib.pyplot as plt

plt.plot(result_random['param_C'],result_random['mean_train_score'],label='Train')
plt.plot(result_random['param_C'],result_random['mean_test_score'],label='Test')
plt.legend()
plt.show()