
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os
import pickle

# -------------------------------------------

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


# -------------------------------------------


#with open('tokenizer.pkl', 'wb') as f:
 #   pickle.dump(tokenizer, f, pickle.HIGHEST_PROTOCOL)

tokenizer = pickle.load( open( "tokenizer.pkl", "rb" ) )

#your_text = ["I have problems with the activation function of my neural network."]

#your_text = tokenizer.texts_to_matrix(your_text, mode='binary')

#data_content_pyso_y = tokenizer.texts_to_matrix(data_content_pyso_y, mode='binary')
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


x_train = tokenizer.texts_to_matrix(x_train, mode='binary')



# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#score = loaded_model.predict_on_batch(your_text)
score = loaded_model.evaluate(x_train, y_train, verbose=2)
print(score)

# Output -------------------

#Using TensorFlow backend.
#2018-02-25 03:11:37.141494: I C:\tf_jenkins\workspace\rel-win\M\windows\PY\36\tensorflow\core\platform\cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
#Loaded model from disk
#[0.29946980278193952, 0.88387000000000004]

#Process finished with exit code 0
