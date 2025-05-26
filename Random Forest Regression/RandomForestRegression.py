import numpy as np
from DecisionTreeRegression import DecisionTreeRegressor
from joblib import Parallel, delayed

class RandomForestRegressor:
    def __init__(self, sample_size, n_trees, max_features = None):
        self.sample_size = sample_size
        self.n_trees = n_trees
        self.max_features = max_features
        self.trees = []
    def fit(self, X, y):
        self.trees = Parallel(n_jobs=-1)(  # -1 = use all cores
        delayed(self.train_single_tree)(X, y) for _ in range(self.n_trees)
    )
    def train_single_tree(self, X, y):
        
        for tree in range(self.n_trees):
            
            indices = np.random.choice(len(X), size = self.sample_size, replace=True)
            X_shuffle = X[indices]
            y_shuffle = y[indices]

            if self.max_features is not None:
                feature_indices = np.random.choice(X.shape[1], self.max_features, replace=True)
                X_shuffle = X_shuffle[:, feature_indices]
            else:
                feature_indices = None  # use all features

            regressor = DecisionTreeRegressor(max_depth=1000, min_num_samples=2)
            regressor.fit(X_shuffle, y_shuffle)
            
            return (regressor, feature_indices)
    
    def predict(self, X):
        predictions = []
        for tree, features in self.trees:
            if features is not None:
                X_input = X[:, features]
            else:
                X_input = X

            predictions.append(tree.predict(X_input))
        
        return np.mean(predictions, axis = 0)




