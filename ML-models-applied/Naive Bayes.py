# Finding Chandler_Naive_Bayes

# Importing the libraries
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

# Creating the Bag Of Words Model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer() 
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values
p=pd.DataFrame(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
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

