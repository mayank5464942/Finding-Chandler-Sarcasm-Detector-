# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 17:18:14 2019

@author: mayank kumar
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset 
dataset = pd.read_csv('train.txt', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 39780):
    text = re.sub('[^a-zA-Z]', ' ', dataset['text'][i])
    text = text.lower()
    text = text.split()
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
    text = ' '.join(text)
    corpus.append(text)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 25000) 
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values
p=pd.DataFrame(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0)
    
embed_dim = 128
lstm_out = 196
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.models import Sequential
classifier = Sequential()
classifier.add(Embedding(25000,196,input_length = X.shape[1]))
classifier.add(SpatialDropout1D(0.4))
classifier.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
classifier.add(Dense(1,activation='softmax'))
classifier.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

batch_size = 32
classifier.fit(X_train, y_train, epochs = 25, batch_size=batch_size, verbose = 2)

y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Calculating F1 Score
from sklearn.metrics import f1_score
print(f1_score(y_test,y_pred))

#Test Dataset
data = pd.read_csv('test.txt', delimiter = '\t', quoting = 2)

corpus1 = []
for i in range(0, 1975):
    text1 = re.sub('[^a-zA-Z]', ' ', data['text'][i])
    text1 = text1.lower()
    text1 = text1.split()
    ps = PorterStemmer()
    text1 = [ps.stem(word) for word in text1 if not word in set(stopwords.words('english'))]
    text1 = ' '.join(text1)
    corpus1.append(text1)
    
Prediction = cv.transform(corpus1).toarray()

answer = classifier.predict(Prediction)

#Converting into CSV file
prediction = pd.DataFrame(answer, columns=['label']).to_csv('Output.csv')


