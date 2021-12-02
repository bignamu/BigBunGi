import numpy as np
import pandas as pd

import sklearn

print(sklearn.__all__)

df = pd.read_csv('house_price.csv',encoding='utf-8')

print(df.shape)
print(df.columns)

from sklearn.model_selection import train_test_split

# help(sklearn.model_selection)

X = df.loc[:,df.columns[:len(df.columns)-1]]
# print(X)
y = df.loc[:,df.columns[-1]]
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

print(y_train.mean(),y_test.mean())

# Normalize

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

minmax = MinMaxScaler()
minmax.fit(X_train)
minmax_X_train = minmax.transform(X_train)
minmax_X_test = minmax.transform(X_test)

stan = StandardScaler()
stan.fit(X_train)
stan_X_train = stan.transform(X_train)
stan_X_test = stan.transform(X_test)


from sklearn.linear_model import LinearRegression

# print(sklearn.linear_model.__all__)

model = LinearRegression()
model2 = LinearRegression()

model.fit(minmax_X_train,y_train)
model2.fit(stan_X_train,y_train)

pred_train = model.predict(minmax_X_train)
mm_score = model.score(minmax_X_train,y_train)
pred_test = model.predict(minmax_X_test)

pred_train2 = model2.predict(stan_X_train)
stan_score = model2.score(stan_X_train,y_train)
pred_test2 = model.predict(stan_X_test)



print(pred_train)
print(mm_score,stan_score)

from sklearn.metrics import mean_squared_error

MSE = mean_squared_error(y_test,pred_test)
print(np.sqrt(MSE))

