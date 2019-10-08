from flask import Flask , redirect, url_for, request, render_template
import keras
from tensorflow.python.client import device_lib
import string
import numpy as np
import pandas as pd
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import re


import sys
import os
from sklearn.manifold import TSNE
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout
from keras.models import Model

from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import string
from matplotlib import pyplot as plt
from keras.layers import Dense, Embedding, LSTM, GRU

import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)

import nltk

import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords

MAX_SEQUENCE_LENGTH = 300
MAX_NB_WORDS = 50000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
from keras.models import load_model

app = Flask(__name__)

@app.route('/',methods = ['POST', 'GET'])
def getRes():
	if request.method == 'POST':
		data = request.form['str']
		return redirect(url_for('results',data = data))
	else:

		return render_template('home.html')

@app.route('/results/<data>')
def results(data):

	data_train = pd.read_csv('clickbait_non-clickbait_data_32000.csv', encoding = 'unicode_escape')


	texts = []
	labels = []
	stop_words = set(stopwords.words('english'))

	for i in range(data_train.text.shape[0]):
	    #text1 = data_train.title[i]
	    text2 = data_train.text[i]
	    text =  str(text2)
	    texts.append(text)
	    labels.append(data_train.label[i])
	text2= []
	for i in texts:

	    word_tokens = word_tokenize(i)

	    filtered_sentence = [w for w in word_tokens if not w in stop_words]

	    filtered_sentence = []

	    for w in word_tokens:
	        if w not in stop_words:
	            filtered_sentence.append(w)
	    filtered_sentence = "".join([" "+j if not j.startswith("'") and j not in string.punctuation else j for j in filtered_sentence]).strip()
	    text2.append(filtered_sentence)
	  

	tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
	tokenizer.fit_on_texts(text2)

	input_text = data
	input_text = tokenizer.texts_to_sequences([input_text])
	input_text = pad_sequences(input_text, maxlen=300)
	model = load_model('lstm.h5')
	res = model.predict_classes(input_text)
	val = list(res)[0]
	return render_template('results.html', val = val, data = data) 

if __name__ == '__main__':
    app.run(threaded=False)