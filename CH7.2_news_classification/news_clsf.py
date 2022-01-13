from tensorflow.keras.datasets import reuters
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt

# 하이퍼 파라미터
word_kind = 1000
len_news = 100
num_rnn = 500
epoch = 30
patience = 5  # 평가 데이터의 오차가 patience 회 연속 증가하면 학습 중단

# 데이터 불러오기
(x_train, y_train), (x_test, y_test) = \
    reuters.load_data(num_words=word_kind, test_split=0.2)

# 뉴스 길이 100으로 맞추기
x_train = pad_sequences(x_train, maxlen=len_news)
x_test = pad_sequences(x_test, maxlen=len_news)

# 원핫 인코딩 적용
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 모델 생성
model = Sequential()
model.add(Embedding(word_kind, num_rnn))
model.add(LSTM(num_rnn))
# model.add(SimpleRNN(num_rnn))
model.add(Dense(y_train.shape[1], activation='softmax'))
model.summary()

# 모델 컴파일 하기
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])

# 조기 종료와 중간 저장
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
mc = ModelCheckpoint(
    'best_model.h5',
    monitor='val_acc',
    mode='max',
    verbose=1,
    save_best_only=True
)

# 학습하기
history = model.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=epoch,
    callbacks=[es, mc],
    validation_data=(x_test, y_test)
)

# 최고 성능 모델 불러오기
loaded_model = load_model('best_model.h5')
print("Test accuracy:", loaded_model.evaluate(x_test, y_test)[1])

epochs = range(len(history.history['acc']))
plt.plot(epochs, history.history['loss'])
plt.plot(epochs, history.history['val_loss'])
plt.title('Learning curve')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'])
plt.show()
