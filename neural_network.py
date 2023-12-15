# -*- coding: utf-8 -*-
"""Neural Network.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xj8cV33iNISQ57mHqerS2VbfrteFpWAf
"""

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from dataset import *

import csv
import pandas as pd

heart_csv = pd.read_csv('heart.csv')
placement_csv = pd.read_csv('Placement_Data_Full_Class.csv')
weather_csv = pd.read_csv('weatherAUS_parsed.csv')
house_csv = pd.read_csv('housePrices.csv')

classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

X = heart_csv.drop(columns=['target'])
y = heart_csv['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

X = placement_csv.drop(columns=['degree_t','salary','gender','ssc_b','hsc_b','hsc_s','workex','specialisation','status'])
y = placement_csv['numerical degree_t']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

X = weather_csv.drop(columns=['Location','MinTemp','MaxTemp','Rainfall','WindGustDir','WindGustSpeed','WindDir9am','WindDir3pm','WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Cloud9am','Cloud3pm','Temp9am','Temp3pm','RainToday','RainTomorrow'])
y = weather_csv['Binary RainTomorrow']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

X = house_csv.drop(columns=['Index','Year','Baths','Beds'])
y = house_csv['Beds']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))