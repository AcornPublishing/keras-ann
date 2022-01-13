from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 하이퍼 파라미터
epoch_num = 1000  # 데이터셋 반복 횟수
learn_rate = 0.01  # 오차에 대한 update 계수
neuron_num = 15  # 은닉층 당 신경 수
reg_val = 0.2  # regularization penalty = 0 or 0.2

# 데이터 불러오기
df = pd.read_csv('sin_noise.csv')
x = df.values[:, 0]
y = df.values[:, 1]

# 모델 생성
model = Sequential()  # 층들이 순차적으로 연결된 신경망 구성
# 첫 번째 은닉층 / 입력층 정보(input_dim)를 제공
model.add(Dense(neuron_num, input_dim=1, activation='tanh'))
# 두 번째 은닉층
model.add(Dense(neuron_num, activation='tanh', kernel_regularizer=l2(reg_val)))
model.add(Dense(1, activation="linear"))  # 출력층
model.summary()  # 층의 순서, 구조, 모수(parameter)의 개수 출력

# 모수 갱신 알고리즘 - Adam 사용
adam_optimizer = Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='mse', optimizer=adam_optimizer)  # 오차 제곱합을 줄이는 모수 갱신

# 모델 학습
hist = model.fit(x, y, verbose=1, epochs=epoch_num)

# 추정
x_new = np.linspace(0, 20, 200)
y_est = model.predict(x_new)

plt.figure(1)
plt.title('Estimation result')
plt.scatter(x, y, label='train dataset')
plt.plot(x_new, y_est, label='fit result')
plt.legend()
plt.show()
