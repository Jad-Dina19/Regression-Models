import numpy as np

class PolynomialRegression:
    def __init__(self):
        
        self.regressor = None

    def fit(self, X, y):
        y = y.ravel()
        self.regressor, *_ = np.linalg.lstsq(X, y, rcond=None)
    
    def OLS_regression(self, X, y):
    
        #OLS closed form formula
        XTX = X.T.dot(X)
        XTX_inverse = np.linalg.pinv(XTX)
        Xy = X.T.dot(y)
        beta = XTX_inverse.dot(Xy)
        return beta
    
    def predict(self, X_poly):
       
        return X_poly.dot(self.regressor)
