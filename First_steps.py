
import keras
import csv
import numpy as np
import pandas as pd
import re


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split

#import data

data_raw_nyt = pd.read_csv("Data/articles1.csv")
data_raw_pyso = pd.read_csv("Data/Answers.csv", encoding = 'iso-8859-1')

data_content_pyso_x = data_raw_pyso['Body'].tolist()[:50000]
data_content_nyt_x = data_raw_nyt['content'].tolist()
data_content_pyso_y = [0]*len(data_content_pyso_x)
data_content_nyt_y = [1]*len(data_content_nyt_x)



def striphtml(data_list):
    p = re.compile(r'<.*?>')
    for i in range(len(data_list)):
        data_list[i] = p.sub('', data_list[i])
    return data_list

print(data_content_pyso_x[:5])
data_content_pyso_x= striphtml(data_content_pyso_x)
print(data_content_pyso_x[:5])




x_train = data_content_pyso_x + data_content_nyt_x
y_train = data_content_pyso_y + data_content_nyt_y




x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.20, random_state=42)


#fix random seed for reproducibility

#load the dataset but only keep the top n words, zero the rest
# # # truncate and pad input sequences
top_words = 100000
max_review_length = 500
x_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_review_length)
# create the model
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(x_train, y_train, nb_epoch=3, batch_size=64)
# # # Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))