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
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)

logreg = LogisticRegression()

logreg.fit(Xtrain, Ytrain)

Ypred = logreg.predict(Xtest)

accuracy = metrics.accuracy_score(Ytest, Ypred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

def predict_diabetes():
    print("Please enter the following values:")
    pregnancies = int(input("Pregnancies: "))
    glucose = int(input("Glucose: "))
    blood_pressure = int(input("Blood Pressure: "))
    skin_thickness = int(input("Skin Thickness: "))
    insulin = int(input("Insulin: "))
    bmi = float(input("BMI: "))
    age = int(input("Age: "))
    pedigree = float(input("Diabetes Pedigree Function: "))

    user_input = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age]], 
                              columns=patientinfo)

    prediction = logreg.predict(user_input)

    if prediction == 1:
        print("Diabetes detected!")
    else:
        print("No Diabetes detected.")
