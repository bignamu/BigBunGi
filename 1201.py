import pandas as pd
import numpy as np

import os

'''
결측값 제거하기
1. df.dropna(axis = 0, how = 'any' or 'all')
aixs : 행(0) or 열(1)
how : any(결측값이 하나라도 있으면 해당하는 행 제거), all(전부다 결측값일때 해당행 제거)
2. df.dropna(thresh = 3)
thresh : 각 행의 결측지가 3개 이상이 되면 삭제
3. df.dropna(subset = [''])
subset : 특정컬럼에서만 결측지가 있는 행 삭제
'''

df = pd.read_csv('Ex_Missing.csv')

colName= df.columns.values
print(colName)

dropped = df.dropna(subset=['salary'])

print(dropped)

