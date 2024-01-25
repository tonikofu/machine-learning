import numpy as np

class KNearestNeighborsModel:
    def __init__(self, n_neighbors = 5):
        self.n_neighbors = n_neighbors
        self.distance = lambda p1, p2: np.sum((p1 - p2) ** 2) ** 0.5
    
    def fit(self, X, Y):
        self.X = X
        self.Y = Y
    
    def predict(self, X):
        return np.array([self.k_neighbors(x) for x in X])

    def k_neighbors(self, x):
        distances = [(self.distance(self.X[i], x), self.Y.iloc[i]) for i in range(self.Y.shape[0])]
        distances.sort(key = lambda x: x[0])

        class_counter = dict()
        
        for i in range(self.n_neighbors):

            if distances[i][-1] in class_counter:
                class_counter[distances[i][-1]] += 1

            else:
                class_counter[distances[i][-1]] = 1
        
        return max(class_counter, key=class_counter.get)
