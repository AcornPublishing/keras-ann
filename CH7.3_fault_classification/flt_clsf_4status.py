from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

import pandas as pd
import numpy as np

# 전압 오차와 상태 레이블 불러오기
df = pd.read_csv('Vdq_err_train.csv')
Verr_train = df.values[:, :-1]
status_train = df.values[:, -1].astype(np.int64) - 1  # 1-15의 상태를 0-14로 변경

# 레이블을 0-3으로 바꾸기 (정상, 스위치 개방, 전류 센서 스케일, 전류 센서 오프셋)
status_train[(status_train > 0) & (status_train <= 6)] = 1
status_train[(status_train > 6) & (status_train <= 10)] = 2
status_train[status_train > 10] = 3

# 레이블을 범주형으로 변환: one-hot 인코딩
status_train_cat = to_categorical(status_train)

# 모델 생성
model = Sequential()
model.add(Dense(16, activation='relu', input_dim=len(Verr_train[0])))
# model.add(Dense(16, activation='relu'))
model.add(Dense(len(status_train_cat[0]), activation='softmax'))
model.summary()

# 모델 컴파일 하기
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 모델 훈련
model.fit(Verr_train, status_train_cat, epochs=5, batch_size=128)

# 평가 데이터셋에서의 정확도 측정하기
df = pd.read_csv('Vdq_err_test.csv')
Verr_test = df.values[:, :-1]
status_test = df.values[:, -1].astype(np.int64) - 1  # 1-15의 상태를 0-14로 변경

# 레이블을 0-3으로 바꾸기 (정상, 스위치 개방, 전류 센서 스케일, 전류 센서 오프셋)
status_test[(status_test > 0) & (status_test <= 6)] = 1
status_test[(status_test > 6) & (status_test <= 10)] = 2
status_test[status_test > 10] = 3

# 레이블을 범주형으로 변환: one-hot 인코딩
status_test_cat = to_categorical(status_test)

test_loss, test_acc = model.evaluate(Verr_test, status_test_cat)
print('test_acc: ', test_acc)
