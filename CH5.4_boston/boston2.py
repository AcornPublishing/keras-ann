from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from sklearn.datasets import load_boston

import matplotlib.pyplot as plt

epoch_num = 500  # 데이터셋 반복 횟수
learn_rate = 0.005  # 오차에 대한 update 계수
neuron_num = 10  # 은닉층 당 신경 수

X, y = load_boston(return_X_y=True)

# (506, 13) : 506개의 데이터(집값), 13개의 특징
X_shape = X.shape

# 마지막 50개의 데이터를 평가용으로 사용
X_train = X[:-50]
y_train = y[:-50]

X_test = X[-50:]
y_test = y[-50:]

# 모델 생성
# 1) 층들이 순차적으로 연결된 신경망 구성
model = Sequential()
# 2-1) 첫째 은닉층 / 신경 10개 / 입력층 정보(X_shape[1]=13) 제공
model.add(Dense(neuron_num, input_dim=X_shape[1], activation="tanh"))
# 2-2) 두번째 은닉층 / 신경 10개
# model.add(Dense(neuron_num, activation='tanh'))
# 3) 출력층
model.add(Dense(1, activation="linear"))

# 층의 순서, 구조, 모수(parameter)의 개수 출력
model.summary()

# 학습 알고리즘과 손실 함수 지정
adam_optimizer = Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss="mse", optimizer=adam_optimizer)  # Mean of Squared Error

# 신경망 학습 & 오차 저감 과정 기록
hist = model.fit(X_train, y_train, epochs=epoch_num)  # 학습 과정 출력
# hist = model.fit(X_train, y_train, verbose=0, epochs=epoch_num)  # 출력 생략

plt.figure(1)
plt.title("Learning Curve")
plt.plot(hist.history["loss"])
plt.show()

# 훈련된 신경망을 사용한 추정 (훈련 데이터셋)
y_est_train = model.predict(X_train)

plt.figure(2)
plt.title("Estimation on Train Dataset")
plt.plot(y_train, label="answer", c="b")  # 정답 (훈련 데이터)
plt.plot(y_est_train, label="estimation", c="r")  # 추정 결과
plt.legend()  # label로 legend 작성
plt.show()

# 훈련된 신경망을 사용한 추정 (평가 데이터셋)
y_est_test = model.predict(X_test)

plt.figure(3)
plt.title("Estimation on Test Dataset")
plt.plot(y_test, label="answer", c="b")  # 정답 (평가 데이터)
plt.plot(y_est_test, label="estimation", c="r")  # 추정 결과
plt.legend()  # label 인자들로 legend(범례) 작성
plt.show()
