import pandas as pd
import numpy as np

df = pd.read_csv('basic1.csv')

df = df[:int(len(df)*0.7)]

print(df.info(),df.isnull().sum())

medi = df['f1'].median()
before = df['f1'].std()
print(df.isnull().sum())

df['f1'] = df['f1'].fillna(medi)
after = df['f1'].std()
print(df.isnull().sum())

print(abs(before-after))
