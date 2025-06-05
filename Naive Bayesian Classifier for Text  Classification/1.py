#Aim :-Assuming a set of documents that need to be classified, use the naïve Bayesian classifier model to perform this task. Built-in Java classes/API can be used to write the program. Calculate the accuracy, precision, and recall for your dataset.

#Program:-

import pandas as pd

# Read the file and split the text and label
with open("dataset.csv", 'r') as file:
    lines = file.readlines()
    
# Process each line to separate message and label
messages = []
labels = []
for line in lines:
    if '->' in line:
        text, label = line.strip().split('->')
        messages.append(text.strip())
        labels.append(label.strip())

# Create DataFrame
msg = pd.DataFrame({
    'message': messages,
    'label': labels
})
print("Total Instances of Dataset: ", msg.shape[0])
msg['labelnum'] = msg.label.map({'pos': 1, 'neg': 0})

X = msg.message
y = msg.labelnum
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)
from sklearn.feature_extraction.text import CountVectorizer

count_v = CountVectorizer()
Xtrain_dm = count_v.fit_transform(Xtrain)
Xtest_dm = count_v.transform(Xtest)

df = pd.DataFrame(Xtrain_dm.toarray(),columns=count_v.get_feature_names_out())
print(df[0:5])

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(Xtrain_dm, ytrain)
pred = clf.predict(Xtest_dm)

for doc, p in zip(Xtrain, pred):
    p = 'pos' if p == 1 else 'neg'
    print("%s -> %s" % (doc, p))
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
print('Accuracy Metrics: \n')
print('Accuracy: ', accuracy_score(ytest, pred))
print('Recall: ', recall_score(ytest, pred))
print('Precision: ', precision_score(ytest, pred))
print('Confusion Matrix: \n', confusion_matrix(ytest, pred))


#DataSet Used:-
# In folder, there is a dataset named "dataset.csv" which contains the following data:



#OutPut:-

""" Total Instances of Dataset:  42
    amazingplace  an  awesome  dance  do  enemy  he  heis  is  juice
0             1   1        0      0   0      0   0     0   0      0
1             0   0        0      0   0      1   0     1   0      0
2             0   1        1      0   0      0   0     0   1      0
3             1   1        0      0   0      0   0     0   0      0
4             0   0        0      1   0      0   0     0   0      0 

    love  my  not  of  place  sworn  taste  this  thisis  to
0     0   0    0   0      0      0      0     0       1   0
1     0   1    0   0      0      1      0     0       0   0
2     0   0    0   0      1      0      0     1       0   0
3     0   0    0   0      0      0      0     0       1   0
4     1   0    0   0      0      0      0     0       0   1

[5 rows x 21 columns]
Thisis an amazingplace -> neg
Heis my sworn enemy -> neg
This is an awesome place -> pos
Thisis an amazingplace -> pos
I love to dance -> neg
Thisis an amazingplace -> pos
I love to dance -> neg
I do not like he taste of this juice -> neg
Heis my sworn enemy -> neg
I do not like he taste of this juice -> neg
This is an awesome place -> neg
Accuracy Metrics:

Accuracy:  1.0
Recall:  1.0
Precision:  1.0
Confusion Matrix:
    [[8 0]
    [0 3]]                          """


#Result:- The above program is executed successfully.