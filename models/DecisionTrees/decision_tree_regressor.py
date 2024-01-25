import numpy as np

class Node:
    def __init__(self, right = None, left = None, feature = None, condition = None, value = None):
        self.right = right
        self.left = left
        self.feature = feature
        self.condition = condition
        self.value = value

class DecisionTreeRegressorModel:
    def __init__(self, max_depth = 5):
        self.max_depth = max_depth
    
    def fit(self, X, Y):
        self.tree = self.build_node(X, Y, 0)
        
    def predict(self, X):
        return np.array([self.walk_on_tree(X.iloc[i], self.tree) for i in range(X.shape[0])])
    
    def build_node(self, X, Y, depth):
        n_features = X.shape[0]
        n_objects = X.shape[1]

        if depth >= self.max_depth or n_objects < 2:
            value = np.sum(Y) / len(Y)
            return Node(value = value)
    
        random_features = np.random.choice(X.columns, n_features)
        opt_gain = 0

        for feature in random_features:
            Xi = np.array(X[feature])
            conditions = np.unique(Xi)

            for condition in conditions:
                information_gain = self.information_gain(Xi, Y, condition)

                if information_gain > opt_gain:
                    opt_feature = feature
                    opt_condition = condition
                    opt_gain = information_gain

        _right, _left = self.split(np.array(X[opt_feature]), opt_condition)
        right = self.build_node(X.iloc[_right], Y.iloc[_right], depth + 1)
        left = self.build_node(X.iloc[_left], Y.iloc[_left], depth + 1)

        return Node(right, left, opt_feature, opt_condition)

    def information_gain(self, Xi, y, condition):
        entropy_before_split = self.entropy(y)

        right, left = self.split(Xi, condition)
        n_right, n_left = len(right), len(left)
        if n_right == 0 or n_left == 0: return 0
        
        n = len(y)
        entropy_right, entropy_left = self.entropy(y.iloc[right]), self.entropy(y.iloc[left])
        entropy_after_split = n_right / n * entropy_right + n_left / n * entropy_left

        return (entropy_before_split - entropy_after_split)
    
    def entropy(self, y):
        probabilities = y.value_counts() / len(y)

        return -np.sum(probabilities * np.log2(probabilities))
    
    def split(self, Xi, condition):
        left = np.argwhere(Xi <= condition).flatten()
        right = np.argwhere(Xi > condition).flatten()

        return right, left

    def walk_on_tree(self, X, tree):
        if tree.value is not None:
            return tree.value

        if X[tree.feature] > tree.condition:
            return self.walk_on_tree(X, tree.right)

        else:
            return self.walk_on_tree(X, tree.left)
