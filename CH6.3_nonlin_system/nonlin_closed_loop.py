from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# FNN or DFNN
est_delta = True  # 변화량을 추정하려면 True, 상태를 곧바로 추정하려면 False

# 하이퍼 파라미터
order = 2
neuron_num = 10
lr = 0.002
epochs = 1000

# 데이터 불러오기 & 훈련 데이터셋 만들기
output = []
input = []

df = pd.read_csv("nonlin_exp.csv", delimiter=",")
data = df.values  # x, u 순서
for i in range(order, len(data)):  # order 번째 행부터 마지막 행까지 for loop
    # 입력에 과거의 상태들과 입력들을 일렬로 추가
    input.append(np.r_[data[i - order : i, 0], data[i - order : i, 1]])
    # 출력에 다음 샘플링의 상태 추가
    if est_delta:
        output.append(data[i, 0] - data[i - 1, 0])
    else:
        output.append(data[i, 0])

input = np.asarray(input)
# MinMaxScaler가 2차원 입력을 받으므로 차원을 증가시켜 사용
output = np.expand_dims(np.asarray(output), axis=1)

print("input.shape =", input.shape)  # (200-order-1, order*2)
print("output.shape =", output.shape)  # (200-order-1, 1)

# 입출력 정규화
input_scaler = MinMaxScaler()
output_scaler = MinMaxScaler()
input_norm = input_scaler.fit_transform(input)
output_norm = output_scaler.fit_transform(output)

# 모델 생성
model = Sequential()  # 층들이 순차적으로 연결된 신경망 구성
# 첫 번째 은닉층 / 입력층 정보(input_dim)를 제공
model.add(Dense(neuron_num, input_dim=order * 2, activation="tanh"))
model.add(Dense(neuron_num, activation="tanh"))  # 두 번째 은닉층
model.add(Dense(1, activation="linear"))  # 출력층
model.summary()  # 층의 순서, 구조, 모수(parameter)의 개수 출력

model.compile(optimizer=Adam(lr=lr), loss="mse")

# 모델 학습
hist = model.fit(input_norm, output_norm, epochs=epochs, verbose=1)

# 학습 곡선
plt.figure()
if est_delta:
    plt.title("DFNN learning curve")
else:
    plt.title("FNN learning curve")
plt.plot(hist.history["loss"])
plt.ylim(0, 5e-4)
plt.show()

# 폐루프 추정
data_len = len(data)
old_progress = 0
x_est = input[:order, 0]
for i in tqdm(range(order, len(data))):
    # 상태는 추정된 최근 order개를 사용
    # 입력은 input 행렬의 i-order번째 행의 order부터 order*2-1 열을 사용
    new_input = np.expand_dims(
        np.r_[x_est[i - order :], input[i - order, order:]], axis=0
    )
    new_input_norm = input_scaler.transform(new_input)
    if est_delta:
        x_est = np.r_[
            x_est,
            x_est[i - 1]
            + output_scaler.inverse_transform(model.predict(new_input_norm))[0],
        ]
    else:
        x_est = np.r_[
            x_est,
            output_scaler.inverse_transform(model.predict(new_input_norm))[0],
        ]
x_est = x_est[order:]

plt.figure()
plt.title("Closed-loop estimation result")
plt.plot(data[order:, 0], label="ori state")
plt.plot(x_est, label="est state")
plt.legend()
plt.show()
