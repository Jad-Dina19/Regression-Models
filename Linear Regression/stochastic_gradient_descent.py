import numpy as np
def sgd(X, y, learning_rate=0.1, epochs=1000, batch_size=1):
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

def predict(X, theta):
    
    return  X * theta[0][0] + theta[1][0]
    