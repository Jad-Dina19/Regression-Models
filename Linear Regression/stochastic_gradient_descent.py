import numpy as np
class LinearRegression:
    def __init__(self, learning_rate, epochs, batch_size):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.regressor = None
    
    def fit(self, X, y):
        self.regressor = self.sgd(X, y, learning_rate=self.learning_rate, epochs=self.epochs, batch_size=self.batch_size)
    
    def sgd(self, X, y, learning_rate=0.1, epochs=1000, batch_size=1):
        #size of the dataset
        m = X.shape[0]
        #write theta 
        theta = np.random.randn(2, 1)   #randomly initialize theta for random guess
        #write the X bias term
        X_bias = np.c_[np.ones([m, 1]), X]

        #shuffle the dataset
        for epoch in range(epochs):
            
            indicies = np.random.permutation(m)
            X_shuffle = X_bias[indicies]
            y_shuffle = y[indicies]

            #write for loop for batch size
            for i in range(0, m, batch_size): #for smaller datasents range(m) is fine
                #find xbatch and ybatch
                X_batch = X_shuffle[i: i+1]
                y_batch = y_shuffle[i: i+1]

                #compute the gradient
                gradient = 2/m * X_batch.T.dot(X_batch.dot(theta) - y_batch)
            
                #update the weights
                theta -= learning_rate * gradient
        
        #return theta
        return theta

    def predict_value(self, X):
        #returns the prediction of one value
        return  X * self.regressor[1][0] + self.regressor[0][0]
    
    def predict(self, X):
        #return y predictions
        return np.c_[np.ones((X.shape[0], 1)), X].dot(self.regressor)

        