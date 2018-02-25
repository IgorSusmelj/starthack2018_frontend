
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

# -------------------------------------------


#with open('tokenizer.pkl', 'wb') as f:
 #   pickle.dump(tokenizer, f, pickle.HIGHEST_PROTOCOL)

tokenizer = pickle.load( open( "tokenizer.pkl", "rb" ) )

your_text = ["How to add variable in Python, Stackoverflow, variable, code, function"]

your_text = tokenizer.texts_to_matrix(your_text, mode='binary')

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
score = loaded_model.predict(your_text)
print(score)