import numpy as np
import itertools

class LinearRegressionModel:
    def __init__(self, l_rate = 1e-4, n_iter = 1000, alpha = 0):
        self.l_rate = l_rate
        self.n_iter = n_iter
        self.alpha = alpha
    
    def fit(self, X, Y):
        n_features = X.shape[1]
        n_objects = Y.shape[0]

        coef = np.random.uniform(0, 1, n_features)

        for _ in itertools.repeat(None, self.n_iter):
            coef -= (2 * self.l_rate * np.dot(-X.T, Y - np.dot(X, coef.T)) + self.alpha * np.sign(coef)) / (n_objects)
        
        self.coef = coef

    def predict(self, X):
        return np.dot(X, self.coef)
