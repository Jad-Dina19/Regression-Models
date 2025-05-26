
import numpy as np
from pprint import pprint
import time
class DecisionTreeRegressor:
    def __init__(self, min_num_samples = 2, max_depth = 5, min_mse_decrease = 1e-7):
        self.min_num_samples = min_num_samples
        self.max_depth = max_depth
        self.min_mse_decrease = min_mse_decrease
        self.tree = None
    
    def fit(self, X, y):
        #builds regression model for specific data
        start_time = time.time()  
        self.tree = self.decision_tree(X, y, depth=0)
        end_time = time.time()  # End timing
        print(f"[DEBUG] build_tree took {end_time - start_time:.4f} seconds")

        #returns tree 
    def get_tree(self):
        return self.tree
        
    def mse(self, y):
       
        num_samples = len(y)

    #check if there are any samples
        if num_samples == 0:
            return 0
    
        #return mean squared error
        return np.mean((y - np.mean(y))**2)

    def best_split(self, X, y):
        
        num_samples, num_features = X.shape
        current_mse = self.mse(y)
        #start at infinity
        best_mse = float('inf')
        best_threshold = None
        best_feature = None
        #find mse for all features
        for feature in range(num_features):
            sorted_indices = X[:, feature].argsort()
            X_sorted = X[sorted_indices]
            y_sorted = y[sorted_indices]

            for i in range(1, num_samples):
                #skip feature if value is the same as last(MSE will produce same result)
                if X_sorted[i, feature] == X_sorted[i-1, feature]:
                    continue
                
                threshold = (X_sorted[i, feature] + X_sorted[(i - 1), feature]) / 2
            
                #split y into left and right using mask
                y_left = y_sorted[:i]
                y_right = y_sorted[i:]

                #calculate mse for split
                left_mse = self.mse(y_left)
                right_mse = self.mse(y_right)

                weighted_mse = (len(y_left) * left_mse + len(y_right) * right_mse) / len(y)
                
                if weighted_mse < best_mse and current_mse - weighted_mse >= self.min_mse_decrease:
                    
                    #replace values with that of min mse
                    best_mse = weighted_mse
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def decision_tree(self, X, y, depth):
        #get feature and threshold
        
        feature, threshold = self.best_split(X, y)
        
        #base case
        if(len(y) < self.min_num_samples or 
           depth >= self.max_depth or 
           feature is None):
            return {
                'type': 'leaf',
                'value': np.mean(y)
            }
        
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return {'type': 'leaf', 'value': np.mean(y)}
        
        #build decision tree using recursion 
        return {
            'type': 'node',
            'feature': feature,
            'threshold': threshold,
            'left': self.decision_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self.decision_tree(X[right_mask], y[right_mask], depth + 1)
        }

    def predict_one(self, x):
        node = self.tree
        
        while node['type'] == 'node':
            
            if x[node['feature']] <= node['threshold']:
                node = node['left']
            else:
                node = node['right']
            
        return node['value']
        
    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])
