from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.utils import to_categorical
from sklearn.metrics import classification_report

import numpy as np

# MNIST 데이터셋 불러오기
(train_images, train_labels), (test_images, test_labels) = \
    fashion_mnist.load_data()
label_texts = [
    "T-shirt/Top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# 입력 정규화 (0~255의 값을 0~1로)
train_images_norm = train_images.astype("float32") / 255
test_images_norm = test_images.astype("float32") / 255

# 입력에 depth 차원 추가하기
train_images_norm = train_images_norm.reshape(
    train_images_norm.shape[0], 28, 28, 1
)
test_images_norm = test_images_norm.reshape(
    test_images_norm.shape[0], 28, 28, 1
)

# 레이블을 범주형으로 변환: one-hot 코딩
train_labels_cat = to_categorical(train_labels)
test_labels_cat = to_categorical(test_labels)

# 모델 생성
model = Sequential()
model.add(
    Conv2D(
        filters=10,
        kernel_size=(5, 5),
        strides=(1, 1),
        input_shape=(28, 28, 1),
        activation="relu",
    )
)
# model.add(Conv2D(32, (5, 5), border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.1))
model.add(Flatten())  # 10 * (28-4)/2 * (28-4)/2 = 1440
model.add(Dense(128, activation="relu"))
model.add(Dense(len(train_labels_cat[0]), activation="softmax"))
model.summary()

# 모델 컴파일 하기
model.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)

# 모델 훈련
model.fit(train_images_norm, train_labels_cat, epochs=5, batch_size=128)

# 평가 데이터셋으로 정확도 측정하기
# 방법 1
# test_loss, test_acc = model.evaluate(test_images_norm, test_labels_cat)
# print('test_acc: ', test_acc)
# 방법 2
test_pred = np.argmax(model.predict(test_images_norm), axis=1)
print(classification_report(test_labels, test_pred))
