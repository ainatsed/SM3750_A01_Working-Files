# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 23:10:21 2023

@author: Destania
"""

import numpy as np
import pandas as pd
import sklearn
import nltk

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from nltk.corpus import stopwords

#stopwords = stopwords.words('english')
#stop_remove = ['i', 'am', 'm', 'is', 'a']
#stop_remove = []

#creating a custom list of stopwords
#s1 = set(stopwords)
#s2 = set(stop_remove)
#stop_list = list(s1.difference(s2))
#print(stop_list)

# read csv into pandas from the working directory
data = pd.read_csv('bitch_dataset.csv', header=None, names=['context', 'quote'])

X = data.context # class
y = data.quote # class labels

# examine the shape
print(data.shape)

# examine first 10 rows
print(data.head(10))

# examine the class distribution
print("\nclass distribution: ")
print(data.context.value_counts())

# convert label to a numerical variable
data['label_num'] = data.context.map({'pos':0, 'neg':1})

# check that conversion worked
print("\n", data.head(10), "\n")

# define X and y for use with Vectorizer
X = data.quote # class
y = data.label_num # class labels
print(X.shape) # (200,)
print(y.shape) # (200,)

# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X_train.shape) # (150,)
print(X_test.shape) # (50,)
print(y_train.shape) # (150,)
print(y_test.shape) # (50,)

# instantiate the vectoriser
vect = CountVectorizer(min_df=1, stop_words='english')
#vect = TfidfVectorizer(min_df=1, stop_words='english')

# learn training data vocabulary, create doc-term matrix
X_train_dtm = vect.fit_transform(X_train) # combine fit and transform into single step

# examine doc-term matrix
print(X_train_dtm.shape) # (150, 640) training samples, training features
print(X_train_dtm)

# transform testing data (using fitted vocabulary) into doc-term matrix
X_test_dtm = vect.transform(X_test)
print(X_test_dtm.shape) # (50, 640) testing samples, training features
print(X_test_dtm)

# import and instantiate a Multinomial Naive Bayes model
nb = MultinomialNB()

y_train=y_train.astype('int')

# train the model using X_train_dtm (timing it with %time)
nb.fit(X_train_dtm, y_train)

# make class predictions for X_test_dtm
y_pred = nb.predict(X_test_dtm)
y_true = np.array(y_test)

X_test = np.array(X_test)

print("\ncorrect predictions:")
for i in range(0, len(y_test)): 
    if (y_true[i] == y_pred[i]): 
        print(i, X_test[i])
        print("true: ", y_true[i], " ; predicted: ", y_pred[i])    
        
print("\nincorrect predictions:")
for i in range(0, len(y_test)): 
    if (y_true[i] != y_pred[i]): 
        print(i, X_test[i])
        print("true: ", y_true[i], " ; predicted: ", y_pred[i])           
    
# create confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# calculate null accuracy based on class distribution
null_accuracy = data.label_num.mean()
print("\nnull accuracy:", null_accuracy)

# calculate accuracy of class predictions
score = metrics.accuracy_score(y_test, y_pred)
print("\naccuracy:", score)