from sklearn import datasets


def get_dataset():
    """
    Load the iris dataset.
    """
    iris = datasets.load_iris()
    return iris.data, iris.target, 3
