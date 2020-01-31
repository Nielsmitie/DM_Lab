import scipy
import os


def get_dataset(name, path="data/"):
    """Loads the skfeature dataset given its name.
    
    Arguments:
        name {str} -- Name of the dataset
    
    Keyword Arguments:
        path {str} -- Path to the datasets (default: {"data/"})
    
    Returns:
        (X, y, n) -- (dataset, labels, #classes)
    """
    path = os.path.join(path, name + ".mat")
    mat = scipy.io.loadmat(path)
    X = mat['X']
    X = X.astype(float)
    y = mat['Y']
    y = y[:, 0]
    num_classes = len(set(y))
    return X, y, num_classes
