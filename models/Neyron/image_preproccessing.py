import numpy as np

def tf_to_ndarray(X):
    _x, _y = [], []

    for images, labels in X:
        for i in range(len(images)):
            _x.append(images[i].numpy().astype("uint8"))
            _y.append(int(labels[i]))

    _x = np.array(_x)
    _y = np.array([_y])

    return _x, _y
    
def to_1D_array(X):
    X_flatten = X.reshape(X.shape[0], -1).T
    return (X_flatten)
