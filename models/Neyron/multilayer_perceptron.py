import numpy as np

class MultilayerPerceptronModel:
    def __init__(self,
                 layers,
                 task = "classify",
                 l_rate = 1e-4,
                 n_epoch = 50,
                 regularize = None,
                 alpha = 0):
        
        self.layers = layers
        self.l_rate = l_rate
        self.n_epoch = n_epoch
        self.alpha = alpha
        self.n_layers = len(self.layers)
        self.coef = [0 for _ in range(self.n_layers)]

        self.regularize = {None : lambda _: 0,
                           "lasso" : lambda w: self.alpha * np.sign(w),
                           "ridge" : lambda w: self.alpha * 2 * w}[regularize]
        
        self._predict = {"classify" : lambda y: np.where(y > 0.5, 1, 0),
                         "regression" : lambda y: y}[task]
    
        self.activation_function = {"linear" : lambda x: x,
                                    "sigmoid" : lambda x: 1 / (1 + np.exp(-x)),
                                    "tanh" : lambda x: np.tanh(x),
                                    "relu" : lambda x: np.where(x > 0, x, 0)}
        
        self.activation_gradient = {"linear" : lambda u: np.ones(u.shape[0]),
                                    "sigmoid" : lambda u: u * (1 - u),
                                    "tanh" : lambda u: 1 - u ** 2,
                                    "relu" : lambda u: np.where(u > 0, 1, 0)}
        
        self.linear_function = lambda w, x: np.dot(x, w)

    def fit(self, X, Y):
        n_objects, n_features = X.shape

        np.random.seed(0)
        self.coef[0] = np.random.uniform(size=(n_features, self.layers[0][0]))

        for i in range(1, self.n_layers):
            self.coef[i] = np.random.uniform(size=(self.layers[i - 1][0], self.layers[i][0]))
        
        for _ in range(self.n_epoch):
            for index in range(n_objects):
                x, y = X[[index]], Y[[index]]
                delta, grad, U = [[0 for _ in range(self.n_layers)] for _ in range(3)]

                Z = self.linear_function(self.coef[0], x)
                U[0] = self.activation_function[self.layers[0][1]](Z)
                    
                for i in range(1, self.n_layers):
                    Z = self.linear_function(self.coef[i], U[i - 1])
                    U[i] = self.activation_function[self.layers[i][1]](Z)
                    
                delta[-1] = (U[-1] - y) * self.activation_gradient[self.layers[-1][1]](U[-1])
                grad[-1] = np.dot(U[-2].T, delta[-1])

                for i in range(self.n_layers - 2, 0, -1):
                    delta[i] = np.dot(delta[i + 1], self.coef[i + 1].T) * self.activation_gradient[self.layers[i][1]](U[i])
                    grad[i] = np.dot(U[i - 1].T, delta[i])

                delta[0] = np.dot(delta[1], self.coef[1].T) * self.activation_gradient[self.layers[0][1]](U[0])
                grad[0] = np.dot(x.T, delta[0])

                for i in range(self.n_layers):
                    self.coef[i] -= self.l_rate * grad[i] - self.regularize(self.coef[i])

    def predict(self, X):
        Z = self.linear_function(self.coef[0], X)
        U = self.activation_function[self.layers[0][1]](Z)

        for i in range(1, self.n_layers):
            Z = self.linear_function(self.coef[i], U)
            U = self.activation_function[self.layers[i][1]](Z)
        
        return self._predict(U)

class ConvolutionalPerceptronModel(MultilayerPerceptronModel):
    def fit(self, X, Y):
        return 0
