# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint 
# Importing the dataset
dataset = pd.read_csv('50_Startups copy.csv')
'''
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
print(X)
'''
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

#encodes + drops dummy variable to rid of linear dependence
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

#make sure every point in matrix is a float
X = np.array(X, dtype=np.float64)
y = np.array(y, dtype=np.float64)

# Training the Decision Tree Regression model on the whole dataset
import DecisionTreeRegressor as ds
regressor = ds.DecisionTreeRegressor()
regressor.fit(X, y)
pprint(regressor.tree)
# Predicting a new result
print(regressor.predict_one([0, 0, 1, 160000, 130000, 300000]))
print(regressor.predict(X))
# Visualising the Decision Tree Regression results (higher resolution)
'''
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
'''