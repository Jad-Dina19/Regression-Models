# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Training the Random Forest Regression model on the whole dataset
from RandomForestRegression import RandomForestRegressor
regressor = RandomForestRegressor(sample_size = 7000, n_trees = 40, max_features=None)
regressor.fit(X_train, y_train)

# Predicting a new result
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
# Visualising the Random Forest Regression results (higher resolution)

from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))

