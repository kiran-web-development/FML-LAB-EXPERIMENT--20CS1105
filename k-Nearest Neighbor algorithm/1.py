#Aim:-Write a program to implement k-Nearest Neighbor algorithm to classify the iris dataset. Print both correct and wrong predictions. Java/Python ML library classes can be used for this problem.

#Program:-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
df = pd.read_csv("k-Nearest Neighbor algorithm\iris_flower_dataset.csv")
print("\nFirst few rows of the dataset:")
print(df.head())

# Prepare features and target
X = df.drop("species", axis=1)
y = df["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_pred
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
classifier2 = KNeighborsClassifier(n_neighbors=20)
classifier2.fit(X_train, y_train)
y_pred2 = classifier2.predict(X_test)
y_pred2
acc2 = accuracy_score(y_test, y_pred2)
print("Accuracy:", acc2)
cm = confusion_matrix(y_test, y_pred)
print(cm)
sns.heatmap(cm, annot=True)
cm2 = confusion_matrix(y_test, y_pred2)
print(cm2)
sns.heatmap(cm2, annot=True)



#DataSet:-
""" In this folder, there is a dataset named "iris_flower_dataset.csv" which contains the following data:"""




#OutPut:-
"""
First few rows of the dataset:
    sepal_length  sepal_width  petal_length  petal_width species
0           5.1          3.5           1.4          0.2  setosa
1           4.9          3.0           1.4          0.2  setosa
2           4.7          3.2           1.3          0.2  setosa
3           4.6          3.1           1.5          0.2  setosa
4           5.0          3.6           1.4          0.2  setosa
Accuracy: 0.8333333333333334
Accuracy: 0.6666666666666666
[[2 0 0]
 [0 2 0]
 [0 1 1]]
[[2 0 0]
 [1 0 1]
 [0 0 2]]
<Axes: >      """



#Result:- The above program is executed successfully.