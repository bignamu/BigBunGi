import pandas as pd
import numpy as np

df = pd.read_csv('basic1.csv')

# 이상치 =  평균 +- 표준편차*1.5

std = df['age'].describe()['std']*1.5
mean = df['age'].mean()
min = mean-std
max = mean+std
print(std,mean,min,max)

answer = df[(df['age']>max) | (df['age']<min)]['age'].sum()
print(answer)


