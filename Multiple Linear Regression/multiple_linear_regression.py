# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from Backward_elimination import MultipleLinearRegression
# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


#make sure every point in matrix is a float
X = np.array(X, dtype=np.float64)
y = np.array(y, dtype=np.float64)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Predicting the Test set results
regressor = MultipleLinearRegression()
regressor.fit(X_train, y_train)

#prints the prediction vs actual values
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))

