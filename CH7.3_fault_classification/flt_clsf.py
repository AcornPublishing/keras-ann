from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.metrics import classification_report

import pandas as pd
import numpy as np

# 전압 오차와 상태 레이블 불러오기
df = pd.read_csv('Vdq_err_train.csv')
Verr_train = df.values[:, :-1]
status_train = df.values[:, -1].astype(np.int64) - 1  # 1-15의 상태를 0-14로 변경

# 불러온 데이터의 마지막 행 출력
print('Verr_train[-1] =', Verr_train[-1])
print('status_train[-1] =', status_train[-1])

# 레이블을 범주형으로 변환: one-hot 인코딩
print("status_train =", status_train)
status_train_cat = to_categorical(status_train)
print("status_train_cat =")
print(status_train_cat)

# 모델 생성
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=len(Verr_train[0])))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(status_train_cat[0]), activation='softmax'))
model.summary()

# 모델 컴파일 하기
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 모델 훈련
model.fit(Verr_train, status_train_cat, epochs=10, batch_size=128)

# 평가 데이터셋에서의 정확도 측정하기
df = pd.read_csv('Vdq_err_test.csv')
Verr_test = df.values[:, :-1]
status_test = df.values[:, -1].astype(np.int64) - 1  # 1-15의 상태를 0-14로 변경

test_pred = np.argmax(model.predict(Verr_test), axis=1)
print(classification_report(status_test, test_pred))
