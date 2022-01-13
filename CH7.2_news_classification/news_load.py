from tensorflow.keras.datasets import reuters
import matplotlib.pyplot as plt
import numpy as np

# 데이터 불러오기, 평가 데이터로 20% 사용
(x_train, y_train), (x_test, y_test) = \
    reuters.load_data(num_words=None, test_split=0.2)

print('Train data #:', len(x_train))
print('Test data # :', len(x_test))

num_class = y_train.max() + 1  # class는 0부터 45, 46개
print('Class #:', num_class)

# 샘플 출력
print(x_train[0])  # 인덱싱된 단어
print(y_train[0])  # 인덱싱된 분류

# 단어 수 분포 확인
lengths = [len(x) for x in x_train]
plt.hist(lengths, bins=100)
plt.xlabel('# of word')
plt.ylabel('# of news')
plt.show()

# 분류 분포 확인
classes, cnt_class = np.unique(y_train, return_counts=True)
plt.bar(classes, cnt_class)
plt.xlabel('class')
plt.ylabel('# of news')
plt.show()

# 문장 복원
word_index = reuters.get_word_index()
index_word = {v: k for k, v in word_index.items()}
# <pad>: padding, 길이를 맞출때 사용하는 토큰
# <sos>: start of sentence, 문장 시작 토큰
# <unk>: unknown, 모델이 인식할 수 없는 토큰
for idx, token in enumerate(("<pad>", "<sos>", "<unk>")):
    index_word[idx] = token
print(' '.join([index_word[index] for index in x_train[0]]))
