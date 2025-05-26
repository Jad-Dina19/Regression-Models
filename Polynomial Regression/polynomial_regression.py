import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from PolynomialRegression import PolynomialRegression
from sklearn.metrics import r2_score

# Load data
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Polynomial transformation
poly_reg = PolynomialFeatures(degree = 4)
X_poly_train = poly_reg.fit_transform(X_train)
X_poly_test = poly_reg.transform(X_test)

# Train model
regressor = PolynomialRegression()
regressor.fit(X_poly_train, y_train)

# Predict
y_pred = regressor.predict(X_poly_test)

# Output
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(-1, 1), y_test.reshape(-1, 1)), axis=1))

# Score
print("R² score:", r2_score(y_test, y_pred))