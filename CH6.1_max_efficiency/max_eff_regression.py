from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from import_data import read_max_eff
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt

# 하이퍼 파라미터
epoch_num = 1000  # 데이터셋 반복 횟수
learn_rate = 0.01  # 오차에 대한 update 계수
neuron_num = 10  # 은닉층 당 신경 수

# 데이터 불러오기
X, Y = read_max_eff()

# 데이터의 각 열 별 정규화 수행
norm_X = MinMaxScaler().fit_transform(X)
y_scaler = MinMaxScaler()
norm_Y = y_scaler.fit_transform(Y)

# 데이터 셔플
random_index = np.arange(X.shape[0])
np.random.shuffle(random_index)

X_train = norm_X[random_index, :]
Y_train = norm_Y[random_index, :]

# 모델 생성
model = Sequential()  # 층들이 순차적으로 연결된 신경망 구성
# 첫 번째 은닉층 / 입력층 크기 기입(input_dim)
model.add(Dense(neuron_num, input_dim=X.shape[1], activation='tanh'))
# model.add(Dense(neuron_num, activation='tanh'))  # 두 번째 은닉층
# model.add(Dense(neuron_num, activation='relu'))  # 세 번째 은닉층
model.add(Dense(Y.shape[1], activation="linear"))  # 출력층
model.summary()  # 층의 순서, 구조, 모수(parameter)의 개수 출력

# 모수 갱신 알고리즘 - Adam 사용
adam_optimizer = Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='mse', optimizer=adam_optimizer)  # 손실 함수로 오차 제곱합 사용

# 학습
hist = model.fit(X_train, Y_train, epochs=epoch_num, validation_split=0.1)

# 추정하기 (정규화된 입력 사용)
norm_Y_est = model.predict(norm_X)
mse = np.average((norm_Y_est - norm_Y)**2)
print('mse =', mse)

plt.figure(1)
plt.title('Learning Curve')
plt.plot(hist.history['loss'], label='train')
plt.plot(hist.history['val_loss'], label='validation')
plt.ylim([0, 0.01])
plt.legend()
plt.show()

# Id, Iq 값으로 보기 위해 inverse transform 수행
Y_est = y_scaler.inverse_transform(norm_Y_est)

ax = plt.figure(2).add_subplot(projection='3d')
plt.title('Estimated Id')
ax.scatter(X[:, 0], X[:, 1], Y[:, 0], label='train')
ax.scatter(X[:, 0], X[:, 1], Y_est[:, 0], label='estimation')
ax.view_init(35, 45)
plt.legend()
plt.show()

ax = plt.figure(3).add_subplot(111, projection='3d')
plt.title('Estimated Iq')
ax.scatter(X[:, 0], X[:, 1], Y[:, 1], label='train')
ax.scatter(X[:, 0], X[:, 1], Y_est[:, 1], label='estimation')
ax.view_init(35, 135)
plt.legend()
plt.show()
