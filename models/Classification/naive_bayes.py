import numpy as np
from math import pi, sqrt

class NaiveBayesModel:
    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.propability = dict(self.Y.value_counts() / self.Y.shape[0])
        self.disperce = lambda x, m: np.sum((x - m) ** 2) / (x.shape[0] - 1)

    def predict(self, X):
        n_features = X.shape[1]
        y_predicted = []

        for x in X:
            probs_for_classes = {}

            for _class in self.propability:
                X_for_class = self.X[self.Y == _class]
                means = np.mean(X_for_class, axis = 0)
                probs_for_classes[_class] = self.propability[_class]

                for i in range(n_features):
                    disperce = self.disperce(X_for_class.T[i], means[i])
                    probs_for_classes[_class] *= np.exp(-((x[i] - means[i]) ** 2) / (2 * disperce)) / (sqrt(2 * pi * disperce))

            y_predicted.append(max(probs_for_classes, key=probs_for_classes.get))

        return np.array(y_predicted)
