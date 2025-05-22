import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from stochastic_gradient_descent import LinearRegression
from OLS import OLS_regression as ols

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')

# Splitting the dataset into the independent variable (X) and dependent variable (y)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state = 0)

#Build regression model
regressor = LinearRegression(0.1, 1000, 1)

#train model with test set
regressor.fit(X_train, y_train)

#predict y_values
y_pred = regressor.predict(X_train)

#predict one value
print(regressor.predict_value(12.5))

# Visualising the Training set results
plt.scatter(X_train,y_train, color = 'red')
plt.plot(X_train, y_pred, color = 'blue')
plt.title('Salary bs Experiemce (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test,y_test, color = 'red')
plt.plot(X_train, y_pred, color = 'blue')
plt.title('Salary bs Experiemce (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
