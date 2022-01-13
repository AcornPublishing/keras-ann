from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.engine.saving import load_model

from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt


# 하이퍼 파라미터
epoch_num = 500  # 데이터셋 반복 횟수
learn_rate = 0.005  # 오차에 대한 update 계수
neuron_num = 10  # 은닉층 당 신경 수

# True 선택 시 새로운 신경망을 생성하고 학습 후 my_model.hdf5로 저장
# False 선택 시 my_model.hdf5를 불러와 추정
do_train = True
# do_train = False

X, y = load_boston(return_X_y=True)

X_shape = X.shape  # (506, 13) : 506개의 데이터(집값), 13개의 특징

# 정규화를 수행하는 scaler model 생성
boston_scaler = MinMaxScaler()
norm_X = boston_scaler.fit_transform(X)  # 데이터의 각 열 별 정규화 수행

if do_train:
    # 마지막 50개의 데이터는 테스트로 사용
    norm_X_train = norm_X[:-50]
    y_train = y[:-50]

    norm_X_test = norm_X[-50:]
    y_test = y[-50:]

    # 모델 생성
    # 1) 층들이 순차적으로 연결된 신경망 구성
    model = Sequential()
    # 2-1) 첫 번째 은닉층 / 신경 10개 / 입력층 정보(X_shape[1]=13)를 제공
    model.add(Dense(neuron_num, input_dim=X_shape[1], activation="tanh"))
    # 2-2) 두 번째 은닉층 / 신경 10개
    model.add(Dense(neuron_num, activation="tanh"))
    # 3) 출력층
    model.add(Dense(1, activation="linear"))

    # 층의 순서, 구조, 모수(parameter)의 개수 출력
    model.summary()

    # 학습 알고리즘과 손실 함수 지정
    adam_optimizer = Adam(
        lr=learn_rate, beta_1=0.9, beta_2=0.999, amsgrad=False
    )
    model.compile(loss="mse", optimizer=adam_optimizer)  # Mean of Squared Error

    # 신경망 학습 & 오차 저감 과정 기록
    hist = model.fit(norm_X_train, y_train, verbose=0, epochs=epoch_num)

    plt.figure(1)
    plt.title("Learning Curve")
    plt.plot(hist.history["loss"])
    plt.show()

    # 훈련된 신경망을 사용한 추정 (훈련 데이터셋)
    y_est_train = model.predict(norm_X_train)

    plt.figure(2)
    plt.title("Estimation on Train Dataset")
    plt.plot(y_train, label="answer", c="b")  # 정답 (훈련 데이터)
    plt.plot(y_est_train, label="estimation", c="r")  # 추정 결과
    plt.legend()  # label로 legend 작성
    plt.show()

    # 훈련된 신경망을 사용한 추정 (평가 데이터셋)
    y_est_test = model.predict(norm_X_test)

    plt.figure(3)
    plt.title("Estimation on Test Dataset")
    plt.plot(y_test, label="answer", c="b")  # 정답 (평가 데이터)
    plt.plot(y_est_test, label="estimation", c="r")  # 추정 결과
    plt.legend()  # label로 legend 작성
    plt.show()

    # 신경망 모델 저장하기 - 구조, 모수 값, 학습 알고리즘 등 저장
    model.save("my_model.hdf5")
else:
    # 신경망 모델 불러오기
    model = load_model("my_model.hdf5")

    # 학습을 정규화된 X로 하였으므로 추정 시에도 정규화된 값을 사용
    y_est = model.predict(norm_X)

    plt.figure()
    plt.title("Estimation on Whole Dataset")
    plt.plot(y, c="b")  # 정답 (전체 데이터)
    plt.plot(y_est, c="r")  # 추정 결과
    plt.show()
