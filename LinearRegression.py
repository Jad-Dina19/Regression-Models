import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import stochastic_gradient_descent 
from OLS import OLS_regression as ols
# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')

# Splitting the dataset into the independent variable (X) and dependent variable (y)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state = 0)

# finding y hat through the sochastic gradient descent
final_theta = stochastic_gradient_descent.sgd(X_train, y_train, learning_rate=0.01, epochs=1000)
y_pred = np.c_[np.ones((X_train.shape[0], 1)), X_train].dot(final_theta)

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


print('Final theta:', final_theta)
prediction = 12 * final_theta[1][0] + final_theta[0][0]
print('Prediction:', prediction)

#prediction using Ordinary Least Squares Method
y_pred = ols(X_train, y_train)

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