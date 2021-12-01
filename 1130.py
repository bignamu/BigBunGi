import pandas as pd
import numpy as np

import matplotlib

import os

df = pd.read_csv('EX_CEOSalary.csv')
colName = df.columns.values
print(df.head(),colName)

asc = df.sort_values(by=[colName[0]],ascending=False).copy();
ten = asc.iloc[:10,0]
asc.iloc[:10,0] = asc.iloc[9,0]

answer = asc[(asc['sales']>=10000)]
answer = answer['salary'].mean()
print(asc.head(20),'\n',ten)

print(answer)

