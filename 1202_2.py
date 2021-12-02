# Gird Search // Random Search

from typing import Final
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
from sklearn.model_selection import GridSearchCV

# help(GridSearchCV)
param_grid = {'C':[0.001,0.01,0.1,1,10,100]}
grid_search = GridSearchCV(LogisticRegression(),param_grid=param_grid,cv=5,return_train_score=True)
grid_search.fit(X_train,y_train)

print(f'Best Parameter : {grid_search.best_params_}')
print(f'Best Cross-Validation Score : {grid_search.best_score_:.3f}')
print(f'Test set Score : {grid_search.score(X_test,y_test):.3f}')

result_grid = pd.DataFrame(grid_search.cv_results_)
print(result_grid)


# import matplotlib.pyplot as plt

# plt.plot(result_grid['param_C'],result_grid['mean_train_score'],label='Train')
# plt.plot(result_grid['param_C'],result_grid['mean_test_score'],label='Test')
# plt.legend()
# plt.show()

Final_model = LogisticRegression(C=10).fit(X_train,y_train)

pred_train = Final_model.predict(X_train)
pred_test = Final_model.predict(X_test)
train_score = Final_model.score(X_train,y_train)
test_score = Final_model.score(X_test,y_test)
print(pred_train,f'{train_score:.3f}')
print(pred_test,f'{test_score:.3f}')

from sklearn.metrics import confusion_matrix,classification_report

confusion_train = confusion_matrix(y_train,pred_train)
cfreport_train = classification_report(y_train,pred_train)
confusion_test = confusion_matrix(y_test,pred_test)
cfreport_test = classification_report(y_test,pred_test)
print(confusion_train)
print(cfreport_train)

print(confusion_test)
print(cfreport_test)
