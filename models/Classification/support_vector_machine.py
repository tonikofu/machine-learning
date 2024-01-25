import numpy as np

class SupportVectorMachineModel:
    def __init__(self, l_rate = 1e-2, n_iter = 25, alpha = 1e-1):
        self.l_rate = l_rate
        self.n_iter = n_iter
        self.alpha = alpha

    def fit(self, X, Y):
        self.Y = np.array(Y * 2 - 1)
        self.X = np.hstack((np.ones((X.shape[0], 1)), X))

        n_coefs = self.X.shape[1]
        n_objects = self.X.shape[0]
        
        coef = np.random.uniform(0, 1, n_coefs)

        for _ in range(self.n_iter):
        
            for i in range(n_objects):
                M = self.Y[i] * np.dot(coef, self.X[i])
    
                if M >= 1:
                    coef -= self.l_rate * self.alpha * coef / n_objects
                else:
                    coef -= self.l_rate * (self.alpha * coef / n_objects - self.Y[i] * self.X[i])
        
        self.coef = coef
    
    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return (np.sign(np.dot(self.coef, X.T)) + 1) / 2
