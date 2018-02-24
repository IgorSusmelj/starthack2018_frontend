
import keras
import csv
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

#import data

data_raw_nyt = pd.read_csv("Data/articles1.csv")
data_raw_twitter = pd.read_csv("Data/twitter.csv")

print(data_raw_twitter.head())

data_content_nyt = data_raw_nyt['content'].tolist()

print(len(data_content_nyt))

x_train = data_content_nyt[:40000]
x_test = data_content_nyt[40000:50000]

y_train = np.ones((40000, 1))
y_test = np.ones((10000, 1))






# #fix random seed for reproducibility
# np.random.seed(7)
# # load the dataset but only keep the top n words, zero the rest
# # # truncate and pad input sequences
# max_review_length = 500
# x_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
# x_test = sequence.pad_sequences(x_test, maxlen=max_review_length)
# # create the model
# embedding_vecor_length = 32
# model = Sequential()
# model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
# # model.add(LSTM(100))
# # model.add(Dense(1, activation='sigmoid'))
# # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# # print(model.summary())
# # model.fit(X_train, y_train, nb_epoch=3, batch_size=64)
# # # Final evaluation of the model
# # scores = model.evaluate(X_test, y_test, verbose=0)
# # print("Accuracy: %.2f%%" % (scores[1]*100))