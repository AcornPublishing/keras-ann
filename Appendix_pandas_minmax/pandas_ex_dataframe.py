import pandas as pd
import numpy as np

# dictionary 변수 생성
data = {
    'Name': ['Jun Lee', 'Daniel', 'Gyu Lim'],
    'Year': [2015, 2016, 2017],
    'Points': [100, 95, 77]
}

# DataFrame으로 변환하기
df = pd.DataFrame(data)
print('df =')
print(df, '\n')

print('df.values =')
print(df.values, '\n')

# index와 columns 출력
print('df.index =', df.index)
print('df.columns =', df.columns, '\n')

# column 1개 뽑기 - series 반환
print("df['Year'] =")
print(df['Year'], '\n')  # df.Year과 동일

# column으로 얻은 series의 index 확인
print("df['Year'].index =")
print(df['Year'].index, '\n')  # df.index와 동일 결과

# row 1개 뽑기 - series 반환
print("df.loc[1, :] =")  # df[1] 혹은 df[1, :]은 오류 발생
print(df.loc[1, :], '\n')

# row로 얻은 series의 index 확인
print("df.loc[1, :].index =")
print(df.loc[1, :].index, '\n')  # df.column와 동일 결과

# 복수의 column 뽑기 - dataframe 반환
print("df[['Points', 'Year']] =")
print(df[['Points', 'Year']], '\n')

# column 연산 및 새 column 추가
df['Rank'] = df['Points'].rank(ascending=False)
df['High Points'] = df['Points'] > 90
print("df =")
print(df)
