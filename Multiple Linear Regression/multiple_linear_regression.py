# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from Backward_elimination import p_value
from Backward_elimination import b_elim
# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

#encodes + drops dummy variable to rid of linear dependence
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop = 'first'), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

#make sure every point in matrix is a float
X = np.array(X, dtype=np.float64)
y = np.array(y, dtype=np.float64)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Predicting the Test set results
y_pred = b_elim(X_train, y_train, X_test)

#prints the prediction vs actual values
for pred, actual in zip(y_pred.flatten(), y_test.flatten()):
    print(f"Predicted: {pred:.2f}, Actual: {actual:.2f}")

#pedict random values (remember to drop first dummy variable)
y_pred = b_elim(X_train, y_train, [[0, 1, 160000, 130000, 300000]])
print(y_pred)