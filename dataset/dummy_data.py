from skfeature.data import *
import scipy


def get_dataset(path):
    num_classes = None
    mat = scipy.io.loadmat(path)
    X = mat['X']
    X = X.astype(float)
    y = mat['Y']
    y = y[:, 0]
    return X, y, num_classes
