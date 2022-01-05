# CNN for the IMDB problem
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten, Input
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

import numpy as np
import data_helpers
from w2v import train_word2vec

np.random.seed(0)

# ---------------------- Parameters section -------------------
#
# Model type. See Kim Yoon's Convolutional Neural Networks for Sentence Classification, Section 3
model_type = "CNN-rand"  # CNN-rand|CNN-non-static|CNN-static

# Data source
data_source = "local_dir"  # keras_data_set|local_dir

# Model Hyperparameters
embedding_dim = 50
filter_sizes = (3, 8)
num_filters = 10
dropout_prob = (0.5, 0.8)
hidden_dims = 50

# Training parameters
batch_size = 64
num_epochs = 10

# Prepossessing parameters
sequence_length = 400
max_words = 5000

# Word2Vec parameters (see train_word2vec)
min_word_count = 1
context = 10

#
# ---------------------- Parameters end -----------------------


def load_data(data_source):
    assert data_source in ["keras_data_set", "local_dir"], "Unknown data source"
    if data_source == "keras_data_set":
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words, start_char=None,
                                                              oov_char=None, index_from=None)

        x_train = sequence.pad_sequences(x_train, maxlen=sequence_length, padding="post", truncating="post")
        x_test = sequence.pad_sequences(x_test, maxlen=sequence_length, padding="post", truncating="post")

        vocabulary = imdb.get_word_index()
        vocabulary_inv = dict((v, k) for k, v in vocabulary.items())
        vocabulary_inv[0] = "<PAD/>"
    else:
        x, y, vocabulary, vocabulary_inv_list = data_helpers.load_data()
        vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}
        y = y.argmax(axis=1)

        # Shuffle data
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x = x[shuffle_indices]
        y = y[shuffle_indices]
        train_len = int(len(x) * 0.9)
        x_train = x[:train_len]
        y_train = y[:train_len]
        x_test = x[train_len:]
        y_test = y[train_len:]

    return x_train, y_train, x_test, y_test, vocabulary_inv


# Data Preparation
print("Load data...")
x_train, y_train, x_test, y_test, vocabulary_inv = load_data(data_source)


if sequence_length != x_test.shape[1]:
    print("Adjusting sequence length for actual size")
    sequence_length = x_test.shape[1]

input_shape = (sequence_length,)

model_input = Input(shape=input_shape)

# create the model
model = Sequential()
model.add(Embedding(max_words, 32, input_length=sequence_length))
model.add(Conv1D(32, 3, padding='same', activation='relu'))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# Fit the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=2, batch_size=128, verbose=2)
# Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))