
import numpy as np
from scipy import stats

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

def p_value(X, y):
    
    #include intercept with X
    X1 = np.c_[np.ones((len(X), 1)), X]

    #get n and p from size of features matrix
    n, p  = X1.shape

    #get y_hat from OLS_regression on X and y
    y_hat, beta = OLS_regression(X, y)
   
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

def b_elim(X, y, X_test, statistical_sig = 0.05):
    #initial p_values
    X_new = X
    X_new_test = X_test
    p_values = p_value(X_new, y)
    count = 0
    while(np.max(p_values) > statistical_sig and count < X_new.shape[1]):
        #get index of max p_value
       
        index = np.argmax(p_values[1:]) 
        #remove feature from index position to get new X
        X_new = np.delete(X_new, index, axis = 1)
        X_new_test = np.delete(X_new_test, index, axis = 1)
        #calculate new p_values
        p_values = p_value(X_new, y)

        count += 1
    
    #get y_hat and beta
    y_hat, beta = OLS_regression(X_new, y)  
    #get y_pred
    X_new_test = np.c_[np.ones((len(X_new_test), 1)), X_new_test ]
    
    return X_new_test.dot(beta)

    






