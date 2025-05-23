import numpy as np
from DecisionTreeRegression import DecisionTreeRegressor

class RandomForestRegressor:
    def __init__(self, sample_size, n_trees, max_features = None):
        self.sample_size = sample_size
        self.n_trees = n_trees
        self.max_features = max_features
        self.trees = []
    
    def fit(self, X, y):
        
        for tree in range(self.n_trees):
            
            indices = np.random.choice(len(X), size = self.sample_size, replace=True)
            X_shuffle = X[indices]
            y_shuffle = y[indices]

            if self.max_features is not None:
                feature_indices = np.random.choice(X.shape[1], self.max_features, replace=False)
                X_sample = X_sample[:, feature_indices]
            else:
                feature_indices = None  # use all features

            regressor = DecisionTreeRegressor()
            regressor.fit(X_shuffle, y_shuffle)
            
            self.trees.append((regressor, feature_indices))
    
    def predict(self, X):
        predictions = []
        for tree, features in self.trees:
            if features is not None:
                X_input = X[:, features]
            else:
                X_input = X

            predictions.append(tree.predict(X_input))
        
        return np.mean(predictions, axis = 0)




