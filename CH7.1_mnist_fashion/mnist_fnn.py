from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.metrics import classification_report

import random

import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt

# MNIST 데이터셋 불러오기
(train_images, train_labels), (test_images, test_labels) = \
    fashion_mnist.load_data()
label_texts = [
    "T-shirt/Top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

print("train_images.shape =", train_images.shape)
print("test_images.shape =", test_images.shape)

fig = plt.figure(figsize=(15, 3))  # inch 단위로 창 크기 조절
fig.subplots_adjust(wspace=0.7)  # subplot 사이의 간격 조절
fig.suptitle("Images and labels")
gs = GridSpec(nrows=1, ncols=10)
for label_num in range(0, 10):
    label_where = random.choice(np.where(train_labels == label_num)[0])
    ax = plt.subplot(gs[label_num])
    ax.imshow(train_images[label_where], cmap="Greys")
    ax.set_title("[" + str(label_num) + "]\n" + label_texts[label_num])
plt.show()

# 입력 정규화 (0~255의 값을 0~1로)
train_images_norm = train_images.astype("float32") / 255
test_images_norm = test_images.astype("float32") / 255

# 입력을 일렬로 세우기 (28x28 행렬을 784행으로 세움)
train_images_norm = train_images_norm.reshape((60000, 28 * 28))
test_images_norm = test_images_norm.reshape((10000, 28 * 28))

# 레이블을 범주형으로 변환: one-hot 코딩
print("train_labels =", train_labels)
train_labels_cat = to_categorical(train_labels)
print("train_labels_cat =")
print(train_labels_cat)
test_labels_cat = to_categorical(test_labels)

# 모델 생성
model = Sequential()
model.add(Dense(256, activation="relu", input_dim=len(train_images_norm[0])))
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

# 2*10개 데이터 추정해보기
fig = plt.figure(figsize=(15, 5))  # inch 단위로 창 크기 조절
fig.subplots_adjust(wspace=0.7)  # subplot 사이의 간격 조절
fig.suptitle("Estimation results")
gs = GridSpec(nrows=2, ncols=10)
for row_iter in range(0, 2):  # 각 레이블 2개씩 추정
    for label_num in range(0, 10):
        label_where = random.choice(np.where(test_labels == label_num)[0])
        pred_class = np.argmax(
            model.predict(np.array([test_images_norm[label_where]]))
        )
        ax = plt.subplot(gs[label_num + row_iter * 10])
        ax.imshow(test_images[label_where], cmap="Greys")
        ax.set_title(
            "[answer]\n"
            + label_texts[label_num]
            + "\n[estimation]\n"
            + label_texts[pred_class]
        )
plt.show()
