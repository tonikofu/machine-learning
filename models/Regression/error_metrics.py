import numpy as np

def MAE(y_true, y_pred):
    n_objects = y_true.shape[0]
    return np.sum(np.absolute(y_true - y_pred)) / n_objects

def MSE(y_true, y_pred):
    n_objects = y_true.shape[0]
    return np.sum((y_true - y_pred) ** 2) / n_objects

def RMSE(y_true, y_pred):
    return MSE(y_true, y_pred) ** 0.5

def MAPE(y_true, y_pred):
    n_objects = y_true.shape[0]
    return np.sum(np.absolute(y_true - y_pred) / y_true) / n_objects

def R2(y_true, y_pred):
    n_objects = y_true.shape[0]
    u = MSE(y_true, y_pred)
    v = np.sum((y_true - y_true.mean()) ** 2) / n_objects
    return 1 - (u / v)