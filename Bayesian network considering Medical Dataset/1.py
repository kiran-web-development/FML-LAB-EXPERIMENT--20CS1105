#Aim :- Write a program to construct a Bayesian network considering medical data. Use this model to demonstrate the diagnosis of heart patients using standard Heart Disease DataSet. You can use Java/Python ML library classes/API.


# Program:-

import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import DiscreteBayesianNetwork # Changed import
from pgmpy.inference import VariableElimination

data = pd.read_csv("Bayesian network considering Medical Dataset\Medical Dataset.csv") # Corrected file path
heart_disease = pd.DataFrame(data)
print(heart_disease)

model = DiscreteBayesianNetwork([ # Changed class name
    ('age', 'Lifestyle'),
    ('Gender', 'Lifestyle'),
    ('Family', 'heartdisease'),
    ('diet', 'cholesterol'), # Corrected spelling
    ('Lifestyle', 'diet'),
    ('cholesterol', 'heartdisease'), # Corrected spelling
    ('diet', 'cholesterol') # Corrected spelling
])

model.fit(heart_disease, estimator=MaximumLikelihoodEstimator)
HeartDisease_infer = VariableElimination(model)

print('For Age enter SuperSeniorCitizen:0, SeniorCitizen:1, MiddleAged:2, Youth:3, Teen:4')
print('For Gender enter Male:0, Female:1')
print('For Family History enter Yes:1, No:0')
print('For Diet enter High:0, Medium:1')
print('for LifeStyle enter Athlete:0, Active:1, Moderate:2, Sedentary:3')
print('for Cholesterol enter High:0, BorderLine:1, Normal:2')

q = HeartDisease_infer.query(variables=['heartdisease'], evidence={
    'age': int(input('Enter Age: ')),
    'Gender': int(input('Enter Gender: ')),
    'Family': int(input('Enter Family History: ')),
    'diet': int(input('Enter Diet: ')),
    'Lifestyle': int(input('Enter Lifestyle: ')),
    'cholesterol': int(input('Enter Cholestrol: ')) # Corrected spelling
})

print(q)
print(q)



#Data Set:-
#here the data set used is saved in this folder as "Medical Dataset.csv".



#Output:-
"""
+-----------------+---------------------+
For Age enter SuperSeniorCitizen:0, SeniorCitizen:1, MiddleAged:2, Youth:3, Teen:4
For Gender enter Male:0, Female:1
For Family History enter Yes:1, No:0
For Diet enter High:0, Medium:1
for LifeStyle enter Athlete:0, Active:1, Moderate:2, Sedentary:3
for Cholesterol enter High:0, BorderLine:1, Normal:2
Enter Age: 2
Enter Gender: 1
Enter Family History: 0
Enter Diet: 1
Enter Lifestyle: 2
Enter Cholestrol: 1
+-----------------+---------------------+
| heartdisease    |   phi(heartdisease) |
+=================+=====================+
| heartdisease(0) |              0.0000 |
+-----------------+---------------------+
| heartdisease(1) |              1.0000 |
+-----------------+---------------------+
+-----------------+---------------------+
| heartdisease    |   phi(heartdisease) |
+=================+=====================+
| heartdisease(0) |              0.0000 |
+-----------------+---------------------+
| heartdisease(1) |              1.0000 |
+-----------------+---------------------+  """


#Result:-The above program is executed successfully.