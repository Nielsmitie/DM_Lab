import scipy
import os


def get_dataset(name, path="data/"):
    """
    This function loads the skfeature dataset given its name.

    Input
    -----
    name: {string}
            name of the dataset
    path: {string}
            path to the dataset

    Output
    ------
    X: {list}
            X values
    y: {list}
            y values
    num_classes: {int}
            the number of classes
    """
    path = os.path.join(path, name + ".mat")
    mat = scipy.io.loadmat(path)
    X = mat['X']
    X = X.astype(float)
    y = mat['Y']
    y = y[:, 0]
    num_classes = len(set(y))
    return X, y, num_classes
