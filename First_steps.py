
import keras
import csv
import numpy as np
import pandas as pd
import re
import h5py
from html.parser import HTMLParser
from keras.preprocessing.text import Tokenizer
import pickle


from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

#import data

data_raw_nyt = pd.read_csv("Data/articles1.csv")
data_raw_pyso = pd.read_csv("Data/Answers.csv", encoding = 'iso-8859-1')

data_content_pyso_x = data_raw_pyso['Body'].tolist()[:50000]
data_content_nyt_x = data_raw_nyt['content'].tolist()
data_content_pyso_y = [0]*len(data_content_pyso_x)
data_content_nyt_y = [1]*len(data_content_nyt_x)

for i in range(len(data_content_pyso_x)):
    data_content_pyso_x[i] = strip_tags(data_content_pyso_x[i])

x_train = data_content_pyso_x + data_content_nyt_x
y_train = data_content_pyso_y + data_content_nyt_y

max_review_length = 500
# We create a tokenizer, configured to only take
# into account the top-1000 most common on words
tokenizer = Tokenizer(num_words=max_review_length)
# This builds the word index
tokenizer.fit_on_texts(x_train)

# This turns strings into lists of integer indices.
#sequences = tokenizer.texts_to_sequences(data_content_pyso_x)

# You could also directly get the one-hot binary representations.
# Note that other vectorization modes than one-hot encoding are supported!

x_train = tokenizer.texts_to_matrix(x_train, mode='binary')
#data_content_pyso_y = tokenizer.texts_to_matrix(data_content_pyso_y, mode='binary')

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f, pickle.HIGHEST_PROTOCOL)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.20, random_state=42)

#fix random seed for reproducibility

#load the dataset but only keep the top n words, zero the rest
# # # truncate and pad input sequences
top_words = 100000
#max_review_length = 500
x_train = sequence.pad_sequences(x_train, maxlen=max_review_length, )
x_test = sequence.pad_sequences(x_test, maxlen=max_review_length)
# create the model
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(x_train, y_train, nb_epoch=1, batch_size=64)
# # # Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
