# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import predict

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
X_poly = predict.polynomial_regression(X, degree = 4)
print(X_poly)
y_pred, beta = predict.OLS_regression(X_poly, y)
print(y_pred)
print(beta)


# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, y_pred, color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

print(predict.predict(predict.polynomial_regression([[6.5]], degree = 4), beta))


