# -*- coding: utf-8 -*-
"""KNN_From_Scratch.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1B7Y2eCEugwgxpal4KkZuCeO1FS0-yW5H
"""

import pandas as pd
import numpy as np
df=pd.read_csv("Social_Network_Ads.csv")

df

df=df.iloc[:,1:]

df

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Gender']=le.fit_transform(df['Gender'])

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=df.iloc[:,0:3].values
x=sc.fit_transform(x)
y=df.iloc[:,-1].values

x

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)

accuracy_score(y_test,knn.predict(x_test))

import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score

class KNN:
    def __init__(self, k):
        self.n_neighbors = k

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        y_pred = []
        for i in x_test:
            distance = []
            for j in self.x_train:
                distance.append(np.sqrt(np.sum((i - j) ** 2)))
            # Get k nearest neighbors
            n_neighbors = sorted(list(enumerate(distance)), key=lambda x: x[1])[:self.n_neighbors]
            label = self.majority_count(n_neighbors)
            y_pred.append(label)
        return np.array(y_pred)

    def score(self, y_test, y_pred):
        return accuracy_score(y_test, y_pred)

    def majority_count(self, neighbors):
        votes = [self.y_train[i[0]] for i in neighbors]
        vote_count = Counter(votes)
        return vote_count.most_common(1)[0][0]

model = KNN(k=5)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Predictions:", y_pred)

model.score(y_test,y_pred)

# The KNN class I made from scratch works the same as the KNN class of sklearn, with an accuracy of 95% for k=5

