import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/content/pima-indians-diabetes-database/diabetes.csv')
df.head(15)

df.columns

patientinfo = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']
result = ['Outcome']

X = df[patientinfo]
Y = df[result]

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

rfc = RandomForestClassifier()
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2)
rfc.fit(Xtrain, Ytrain)

Ypred = rfc.predict(Xtest)

accuracy = metrics.accuracy_score(Ytest, Ypred)
accuracy
