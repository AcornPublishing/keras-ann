from keras_preprocessing.text import Tokenizer

sentences = [
    'I love keras.',
    'I love you!',
    'What about YOU?'
]

new_sentence = ['He does not like keras.']

tokenizer = Tokenizer(oov_token='Unseen')
tokenizer.fit_on_texts(sentences)
print(tokenizer.word_index)
print(tokenizer.texts_to_sequences(sentences))
print(tokenizer.texts_to_sequences(new_sentence))


def one_hot_encoding(word, word_index):
    one_hot_vector = [0] * len(word_index)
    one_hot_vector[word_index[word]] = 1
    return one_hot_vector


print(one_hot_encoding('keras', tokenizer.word_index))
