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
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2)

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score

models = {
    'LogisticRegression': LogisticRegression(),
    'GaussianNB': GaussianNB(),
    'SVC': SVC(),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'ExtraTreeClassifier': ExtraTreeClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'BaggingClassifier': BaggingClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier()
}

max_accuracy = 0
best_model = None

for name, model in models.items():
    model.fit(Xtrain, Ytrain)
    Ypred = model.predict(Xtest)
    accuracy = accuracy_score(Ytest, Ypred)
    print(f"{name} model with accuracy: {accuracy}")
    
    if accuracy > max_accuracy:
        max_accuracy = accuracy
        best_model = name

print(f"The best model is {best_model} with an accuracy of {max_accuracy}")
