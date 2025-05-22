# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PolynomialRegression import PolynomialRegression

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
#create model
regressor = PolynomialRegression(degree = 4)

#train model on dataset
regressor.fit(X, y)

#predict y values
y_pred = regressor.predict(X)

print(regressor.predict([10]))

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, y_pred, color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()



