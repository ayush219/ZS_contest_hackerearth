#Libraries
import numpy as np
import pandas as pd

#Dataset
dataset_train= pd.read_csv('train_file.csv')
dataset_test= pd.read_csv('test_file.csv')

#cleaning_train
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
corpus_train=[]

for i in range(0,55932):    
    headline=re.sub('[^a-zA-Z]',' ', dataset_train['Headline'][i])
    headline=headline.lower()
    headline=headline.split()
    headline=[ps.stem(word) for word in headline if not word in set(stopwords.words('english'))]
    headline=' '.join(headline)
    corpus_train.append(headline)

#Bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X_train=cv.fit_transform(corpus_train).toarray()
Y2_train=dataset_train.iloc[:,10].values

#Regressor
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y2_train) 

#cleaning_test
corpus_test=[]

for i in range(0,37288):    
    headline=re.sub('[^a-zA-Z]',' ', dataset_test['Headline'][i])
    headline=headline.lower()
    headline=headline.split()
    headline=[ps.stem(word) for word in headline if not word in set(stopwords.words('english'))]
    headline=' '.join(headline)
    corpus_test.append(headline)

#Bag of words
X_test=cv.fit_transform(corpus_test).toarray()

#Prediction
Y2_pred= regressor.predict(X_test)