import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error

url = 'https://media.geeksforgeeks.org/wp-content/uploads/20240320114716/data_for_lr.csv'
df = pd.read_csv(url)

data.head(100)

X = df[['x']].values
Y = df['y'].values

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = 0.2, random_state=650)
linreg = LinearRegression()

print(f"Shape of X: {X.shape}")
print(f"Shape of y: {Y.shape}")

print(f"Shape of Xtrain: {Xtrain.shape}")
print(f"Shape of Ytrain: {Ytrain.shape}")
print(f"Shape of Xtest: {Xtest.shape}")
print(f"Shape of Ytest: {Ytest.shape}")

Ytrain = Ytrain.reshape(-1)
Ytest = Ytest.reshape(-1)

if np.any(np.isnan(Xtrain)):
    print("Warning: Xtrain contains NaN values.")
    Xtrain = np.nan_to_num(Xtrain)

if np.any(np.isnan(Ytrain)):
    print("Warning: Ytrain contains NaN values.")
    Ytrain = np.nan_to_num(Ytrain)

linreg.fit(Xtrain, Ytrain)

Ypred = linreg.predict(Xtest)

if np.any(np.isnan(Ytest)):
    print("Warning: Ytest contains NaN values.")
    Ytest = np.nan_to_num(Ytest)

if np.any(np.isnan(Ypred)):
    print("Warning: Ypred contains NaN values.")
    Ypred = np.nan_to_num(Ypred)

mse = mean_squared_error(Ytest, Ypred)
print(f"Mean Squared Error: {mse}")

r2_score = linreg.score(Xtest, Ytest)
print(f"R² Score (Accuracy): {r2_score}")

plt.scatter(Xtest, Ytest, color='blue', label='Actual data')
plt.plot(Xtest, Ypred, color='red', label='Linear regression line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression on Test Data')
plt.legend()
plt.show()

user_input = float(input("Enter a value for x to predict y: "))
predicted_y = linreg.predict([[user_input]])
print(f"The predicted value of y for x = {user_input} is: {predicted_y[0]}")

r2_score = linreg.score(Xtest, Ytest)
print(f"R² Score (Accuracy): {r2_score}")
