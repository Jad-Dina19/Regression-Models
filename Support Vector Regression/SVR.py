import numpy as np
import time
class SVR():

    def __init__(self, C = 1.0, learning_rate = 0.001, epochs = 100, gamma = 0.5, epsilon = 0.1):
        self.C = C
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.gamma = gamma
        self.epsilon = epsilon
        self.delta_alpha = None
        self.b = None
        self.X_train = None

    def fit(self, X, y):
        self.X_train = X
        alpha, alpha_star = self.sga(X, y)
        self.delta_alpha = alpha - alpha_star
        self.b = self.bias(alpha, alpha_star, X, y)

    def rbf_kernel(self, x1, x2):
        x1 = np.asarray(x1).flatten()  # Ensure 1D
        x2 = np.asarray(x2).flatten()  # Ensure 1D
        diff = x1 - x2
        norm_squared = np.dot(diff, diff)
        return np.exp(-self.gamma * norm_squared)
    
    def kernel_matrix_cross(self, X_test):
        X_train = np.asarray(self.X_train)
        X_test = np.asarray(X_test)

        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

        # Squared norms for training and test sets
        sq_norms_train = np.sum(X_train ** 2, axis=1).reshape(-1, 1)  # (n_train, 1)
        sq_norms_test = np.sum(X_test ** 2, axis=1).reshape(1, -1)    # (1, n_test)

        # Compute squared distances matrix
        dists = sq_norms_train + sq_norms_test - 2 * X_train.dot(X_test.T)  # (n_train, n_test)

        # Apply RBF formula
        K_cross = np.exp(-self.gamma * dists)

        return K_cross

    def rbf_kernel_vector(self, X):
        X = np.asarray(X)
    
        # Squared norms of each row vector
        sq_norms = np.sum(X ** 2, axis=1).reshape(-1, 1)  # Shape (n, 1)
        
        # Compute squared Euclidean distance matrix using broadcasting
        dists = sq_norms + sq_norms.T - 2 * np.dot(X, X.T)  # Shape (n, n)
        
        # Apply the RBF formula
        K = np.exp(-self.gamma * dists)
        
        return K

    def sga(self, X, y):
        n_samples, n_features = X.shape

        alpha = np.zeros(n_samples)
        alpha_star = np.zeros(n_samples)
        
        start = time.time()
        K = self.rbf_kernel_vector(X)
       
        for epoch in range(self.epochs):
            start = time.time()
            for i in range(n_samples):
                s_i = np.dot(alpha - alpha_star, K[:, i])

                grad_alpha = float(y[i] - self.epsilon - s_i)
                grad_alpha_star = float(-y[i] - self.epsilon + s_i)
                
                alpha[i] = np.clip(alpha[i] + self.learning_rate * grad_alpha, 0, self.C)
                alpha_star[i] = np.clip(alpha_star[i] + self.learning_rate * grad_alpha_star, 0, self.C)
             
            print(f"{epoch} epoch {time.time() - start:.2f} seconds")
        return alpha, alpha_star

    def bias(self, alpha, alpha_star, X, y):

        delta_alpha = alpha - alpha_star
        n = len(X)
        
        K = self.rbf_kernel_vector(X)  # Precompute kernel matrix once
        
        b_values = []
        for i in range(n):
            if 0 < alpha[i] < self.C or 0 < alpha_star[i] < self.C:
                f_xi = np.dot(delta_alpha, K[:, i])
                
                if 0 < alpha[i] < self.C:
                    b_i = y[i] - f_xi - self.epsilon
                    b_values.append(b_i)
                elif 0 < alpha_star[i] < self.C:
                    b_i = y[i] - f_xi + self.epsilon
                    b_values.append(b_i)

        return np.mean(b_values) if b_values else 0.0
        
    
    def predict(self, X):
        K_test = self.kernel_matrix_cross(X)
        print(self.b)
        return np.dot(self.delta_alpha, K_test) + self.b