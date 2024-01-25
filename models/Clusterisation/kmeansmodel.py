import numpy as np

class KMeansModel:
    def __init__(self, n_clusters = 2, n_features = 2, n_iter = 100):
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.n_iter = n_iter
        self.centroids_objects = { i : [] for i in range(n_clusters) }
        
        # Случайным образом назначаем центроид каждому кластеру
        self.cluster_centers_ = np.array([[np.random.uniform() for _ in range(n_features)] for _ in range(n_clusters)])

    def fit(self, X):
        n_objects = X.shape[0]
        distances_to_centroid = np.array([0. for _ in range(self.n_clusters)])
        self.labels_ = np.array([0 for _ in range(n_objects)])

        for _ in range(self.n_iter):
            for xi in range(n_objects):
                x = X[xi]

                # Находим евклидово расстояние от объекта до каждого центроида
                for centroid_idx in range(self.n_clusters):
                    distance = np.sum((x - self.cluster_centers_[centroid_idx]) ** 2) ** 0.5
                    distances_to_centroid[centroid_idx] = distance

                # Закрепляем объект за ближайшим центроидом
                label = np.argmin(distances_to_centroid)
                self.centroids_objects[label].append(x)
                self.labels_[xi] = label

            # Находим новое метоположение центроида, взяв среднее значение всех наблюдений
            for i in range(self.n_clusters):
                for j in range(self.n_features):
                    centroid = self.centroids_objects[i]
                    self.cluster_centers_[i][j] = np.sum(np.array(centroid)[:, j]) / len(centroid)
