from sklearn import datasets


def get_dataset():
    iris = datasets.load_iris()
    return iris.data, iris.target, 3
