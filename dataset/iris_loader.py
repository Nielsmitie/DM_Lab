from sklearn import datasets


def get_dataset():
    """Loads the iris dataset.

    Returns:
        (X, y, n) -- (dataset, labels, #classes)
    """
    iris = datasets.load_iris()
    return iris.data, iris.target, 3
