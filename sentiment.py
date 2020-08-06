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
corpus_train_title=[]
corpus_train_headline=[]

for i in range(0,55932):    
    title=re.sub('[^a-zA-Z]',' ', dataset_train['Title'][i])
    title=title.lower()
    title=title.split()
    title=[ps.stem(word) for word in title if not word in set(stopwords.words('english'))]
    title=' '.join(title)
    corpus_train_title.append(title)
    
    headline=re.sub('[^a-zA-Z]',' ', dataset_train['Headline'][i])
    headline=headline.lower()
    headline=headline.split()
    headline=[ps.stem(word) for word in headline if not word in set(stopwords.words('english'))]
    headline=' '.join(headline)
    corpus_train_headline.append(headline)

#Bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)

X_train_title=cv.fit_transform(corpus_train_title).toarray()
Y_train_title=dataset_train.iloc[:,9].values

X_train_headline=cv.fit_transform(corpus_train_headline).toarray()
Y_train_headline=dataset_train.iloc[:,10].values

#Regressor
from sklearn.linear_model import LinearRegression

regressor1=LinearRegression()
regressor1.fit(X_train_title,Y_train_title) 

regressor2=LinearRegression()
regressor2.fit(X_train_headline,Y_train_headline) 

#cleaning_test
corpus_test_title=[]
corpus_test_headline=[]

for i in range(0,37288):    
    title=re.sub('[^a-zA-Z]',' ', dataset_test['Title'][i])
    title=title.lower()
    title=title.split()
    title=[ps.stem(word) for word in title if not word in set(stopwords.words('english'))]
    title=' '.join(title)
    corpus_test_title.append(title)   

    headline=re.sub('[^a-zA-Z]',' ', dataset_test['Headline'][i])
    headline=headline.lower()
    headline=headline.split()
    headline=[ps.stem(word) for word in headline if not word in set(stopwords.words('english'))]
    headline=' '.join(headline)
    corpus_test_headline.append(headline)

#Bag of words
X_test_title=cv.fit_transform(corpus_test_title).toarray()
X_test_headline=cv.fit_transform(corpus_test_headline).toarray()

#Prediction
Y_pred_title= regressor1.predict(X_test_title)
Y_pred_headline= regressor2.predict(X_test_headline)

#Submission
s=pd.DataFrame()
s['IDLink']=dataset_test['IDLink']
s['SentimentTitle']=Y_pred_title
s['SentimentHeadline']=Y_pred_headline
sub=s.to_csv('Submission.csv',index=None)