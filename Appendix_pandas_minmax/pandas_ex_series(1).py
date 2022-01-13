import pandas as pd

# series 변수
sr = pd.Series([1, 3, -5, -7])
print('sr =')
print(sr, '\n')
print('sr.values =', sr.values)
print('sr.index =', sr.index, '\n')

sr.index = ['a', 'b', 'd', 'c']
print('sr =')
print(sr, '\n')

# loc 을 사용해 특정 index들의 value 얻기 - 순서 바꾸기
print("sr.loc['d', 'a'] =")
print(sr.loc[['d', 'a']])
