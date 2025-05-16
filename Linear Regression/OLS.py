import pandas as pd
import numpy as np

def OLS_regression(X, y):
   #Set x properly for fomula
   m = len(X)
   X1 = np.c_[np.ones((m, 1)), X]
   
   #OLS closed form formula
   XTX = X1.T.dot(X1)
   XTX_inverse = np.linalg.inv(XTX)
   Xy = X1.T.dot(y)
   beta = XTX_inverse.dot(Xy)
   return X1.dot(beta)