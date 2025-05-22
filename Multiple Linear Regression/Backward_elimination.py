import numpy as np
from scipy import stats

class MultipleLinearRegression:

    def __init__(self, statistical_sig):
        self.statistical_sig = statistical_sig
        self.regressor = None
        self.removed = None
    
    def fit(self, X_train, y):
        X_new, self.removed = self.b_elim(X_train, y, statistical_sig=self.statistical_sig)
        self.regressor = self.OLS_regression(X_new, y)  

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

    def p_value(self, X, y):
        
        #include intercept with X
        X1 = np.c_[np.ones((len(X), 1)), X]

        #get n and p from size of features matrix
        n, p  = X1.shape

        #get y_hat from OLS_regression on X and y
        beta = self.OLS_regression(X, y)

        y_hat = X1.dot(beta)
        #calculate residuals
        r = y - y_hat
        
        #calculate residual mean square error
        s = r.T.dot(r)/(n-p)
        
        #calculate standard error
        s_e = np.sqrt(np.diag(s*np.linalg.pinv(X1.T.dot(X1))))
    
        #calculate t-statistic
        t_stat = beta.flatten()/s_e
        
        #calculate p_values
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stat), df =(n - p)))

        return p_values

    def b_elim(self, X, y, statistical_sig = 0.05):
        #initial p_values
        X_new = X
        p_values = self.p_value(X_new, y)
        count = 0
        
        remaining_indices = list(range(X_new.shape[1]))

        while(np.max(p_values) > statistical_sig and count < X_new.shape[1]):
            #get index of max p_value
            index = np.argmax(p_values[1:])

            #remove feature from index position to get new X
            X_new = np.delete(X_new, index, axis = 1)

            #delete index
            del remaining_indices[index]
            #calculate new p_values
            p_values = self.p_value(X_new, y)

            count += 1
        
        #get y_hat and beta
        return X_new, remaining_indices

    def predict(self, X):
        #make np array
        X = np.array(X)

        #make sure dimension permits multiplication
        if X.ndim == 1:
            
            X = X.reshape(1, -1)
        #removes statistically insignificant features
        X = X[:, self.removed]

        X = np.c_[np.ones((len(X), 1)), X]
        
        return X.dot(self.regressor)

   