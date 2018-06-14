#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 18:13:45 2018

@author: spiky
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

#importing dataset
training_dataset = pd.read_csv('training.tsv', delimiter = '\t', quoting = 3)
test_dataset = pd.read_csv('testdata.tsv', delimiter = '\t', quoting = 3)

#cleaning text
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,7086):
    Review = re.sub('[^a-zA-Z]',' ', training_dataset['review'][i])
    Review = Review.lower()
    Review = Review.split()
    ps = PorterStemmer()
    Review = [ps.stem(word) for word in Review if not word in set(stopwords.words('english'))]
    Review = ' '.join(Review)
    corpus.append(Review)

#creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = training_dataset.iloc[:, 0].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

#calculating accuracy 
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
print("SVM AUC: {0}".format(metrics.auc(fpr, tpr)))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


tr = []
for i in range(0,33052):
    mr = test_dataset['review'][i]
    tr.append(mr)
    
test_set_features=cv.transform(tr).toarray()
predictions2=classifier.predict(test_set_features)
df=pd.DataFrame(data={"Predicted Score":predictions2,"Text":tr})


df.to_csv("./SVM_predictions.csv",sep=',',index=False)