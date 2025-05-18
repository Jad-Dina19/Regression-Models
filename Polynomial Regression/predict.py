import numpy as np


def OLS_regression(X, y):
  
   #Set x properly for fomula
   m = len(X)
   X1 = np.c_[np.ones((m, 1)), X]
   
   #OLS closed form formula
   XTX = X1.T.dot(X1)
   XTX_inverse = np.linalg.pinv(XTX)
   Xy = X1.T.dot(y)
   beta = XTX_inverse.dot(Xy)
   return X1.dot(beta), beta

def  polynomial_regression(X, degree = 1):
    X = np.array(X)
    X_new = X
    for i in range(2,degree+1):
        X_new = np.c_[X_new, X**i]
    return X_new

def predict(X_poly, beta):
    X_poly = np.c_[np.ones((X_poly.shape[0], 1)), X_poly]
    return X_poly.dot(beta)
