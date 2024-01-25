import numpy as np
from sklearn.preprocessing import StandardScaler

'''
1. Стандартизация данных (итерируемся по столбцам, от каждого элемента столбца отнимаем среднее
по столбцу и делим на отклонение столбца - то, что делает StandartScaller).
2. Вычисление ковариационной матрицы.
3. Вычисление собственных векторов и собственных значений ковариационной матрицы.
4. Сортировка пар <собственное значение, собственный вектор> по убыванию.
5. Выбор первых k пар <собственное значение, собственный вектор>, где k - размерность целевого
пространства.
6. Матрица, составленная из k собственных векторов - матрица преобразования из данного
пространства в пространство с размерностью k.
7. Чтобы произвести понижение размерности необходимо умножить матрицу стандартизированных
входных данных (результат пункта 1) на матрицу из k собственных векторов (результат пункта 6).
'''

class PCAModel:
    def __init__(self, n_components = 2):
        self.n_components = n_components
    
    def fit_transform(self, X):
        X = StandardScaler().fit_transform(X)

        transform_X = []
        cov = np.cov(X.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        eig = { eigenvalues[i] : eigenvectors[i] for i in range(eigenvalues.shape[0]) }
        eig_sorted = sorted(eig.items(), reverse=True)

        for i in range(self.n_components):
            transform_X.append(eig_sorted[i][1])
        
        transform_X = np.array(transform_X)
        return np.dot(X, transform_X.T)
