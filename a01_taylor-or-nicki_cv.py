# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# read csv into pandas from the working directory
data = pd.read_csv('taylor-or-nicki.csv', header=None, names=['artist', 'lyric'])

X = data.artist # class
y = data.lyric # class labels

# examine the shape
print(data.shape)

# examine first 10 rows
print(data.head(10))

# examine the class distribution
print("\nclass distribution: ")
print(data.artist.value_counts())

# convert label to a numerical variable
data['label_num'] = data.artist.map({'swift':0, 'nicki':1})

# check that conversion worked
print("\n", data.head(10), "\n")

# define X and y for use with COUNTVECTORIZER
X = data.lyric # class
y = data.label_num # class labels
print(X.shape) # (100,) vectorising will create 2D
print(y.shape) # (100,)

# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X_train.shape) # (75,)
print(X_test.shape) # (25,)
print(y_train.shape) # (75,)
print(y_test.shape) # (25,)

# instantiate the vectoriser
vect = CountVectorizer(min_df=1, stop_words='english')

# learn training data vocabulary, create doc-term matrix
#vect.fit(X_train)
#X_train_dtm = vect.transform(X_train)
X_train_dtm = vect.fit_transform(X_train) # combine fit and transform into single step

# examine doc-term matrix
print(X_train_dtm.shape) # (75, 487) training samples, training features
print(X_train_dtm)

# transform testing data (using fitted vocabulary) into doc-term matrix
X_test_dtm = vect.transform(X_test)
print(X_test_dtm.shape) # (25, 487) testing samples, training features
print(X_test_dtm)

# import and instantiate a Multinomial Naive Bayes model
nb = MultinomialNB()

# train the model using X_train_dtm (timing it with %time)
nb.fit(X_train_dtm, y_train)

# make class predictions for X_test_dtm
y_pred = nb.predict(X_test_dtm)
y_true = np.array(y_test)
for i in range(0, len(y_test)): 
    print("true: ", y_true[i], " ; predicted: ", y_pred[i])

# create confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
# plot heatmap
#sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=False,
            #xticklabels=y_train.lyric,
            #yticklabels=y_train.lyric)
#plt.xlabel('true')
#plt.ylabel('predicted')

# calculate accuracy of class predictions
score = metrics.accuracy_score(y_test, y_pred)
print("\naccuracy:", score)








