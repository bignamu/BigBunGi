import pandas as pd
import numpy as np

df = pd.read_csv('basic1.csv')

# print(df.shape,df.info(),df.describe(),df.head())

df = df.sort_values(by='f5',ascending=False)

min_val = df['f5'][:10].min()
# print('fdsafasdfdasfdsa',min_val)
df['f5'][:10] = min_val

# print(df.head(10))

df_new = df[df['age']>=80].copy()
answer = df_new['f5'].mean()

print(answer)