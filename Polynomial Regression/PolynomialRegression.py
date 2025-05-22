import numpy as np

class PolynomialRegression:
    def __init__(self, degree):
        self.degree = degree
        self.regressor = None

    def fit(self, X, y):
        self.regressor = self.OLS_regression(self.polynomial_matrix(X, self.degree), y)
    
    def OLS_regression(self, X, y):
    
        #Set x properly for fomula
        m = len(X)
        X1 = np.c_[np.ones((m, 1)), X]
        
        #OLS closed form formula
        XTX = X1.T.dot(X1)
        XTX_inverse = np.linalg.pinv(XTX)
        Xy = X1.T.dot(y)
        beta = XTX_inverse.dot(Xy)
        return beta

    def  polynomial_matrix(self, X, degree = 1):
        # make an array
        X = np.array(X)
        X_new = X
        
        #concatenate column power matricies onto original matrix
        
        for i in range(2,degree+1):
           
            X_new = np.c_[X_new, X**i]
        
        return X_new

    def predict(self, X_poly):
       #make an array
        X_poly = np.array(X_poly)

        #check n dimension for multiplication 
        if X_poly.ndim == 1:
            X_poly = X_poly.reshape(1, -1)

        #create poly matrix
        X_poly = self.polynomial_matrix(X_poly, degree = self.degree)
        
        #calculate y pred
        X_poly = np.c_[np.ones((X_poly.shape[0], 1)), X_poly]
        return X_poly.dot(self.regressor)
