!kaggle datasets download uciml/pima-indians-diabetes-database

!unzip pima-indians-diabetes-database.zip -d pima-indians-diabetes-database

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

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC

Ypred = svm.predict(Xtest)

accuracy = metrics.accuracy_score(Ytest, Ypred)
accuracy
