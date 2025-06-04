#Aim:- Write a Program to demonstrate the working of the decision tree algorithm.

#Program:-

#Three lines to make our compiler able to draw:
import sys
import matplotlib
matplotlib.use('Agg')
import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
df = pandas.read_csv("dataset.csv")
d = {'UK': 0, 'USA': 1, 'N': 2}
df['Nationality'] = df['Nationality'].map(d)
d = {'YES': 1, 'NO': 0}
df['Go'] = df['Go'].map(d)
features = ['Age', 'Experience', 'Rank', 'Nationality']
X = df[features]
y = df['Go']
clf = DecisionTreeClassifier()
clf = clf.fit(X, y)
plt.figure(figsize=(10, 6))
tree.plot_tree(clf, feature_names=features, class_names=['NO', 'YES'], filled=True)
plt.savefig('decision_tree.png')
plt.close()


#DataSet Used:-

#In folder, there is a dataset named "dataset.csv" which contains the following data:
#Age,Experience,Rank,Nationality,Go


#Result:-The above program is executed successfully.


