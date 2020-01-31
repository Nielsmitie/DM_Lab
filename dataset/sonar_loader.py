from scipy.io.arff import loadarff
import os


def get_dataset():
    """Load the sonar dataset.
    https://datahub.io/machine-learning/sonar#resource-sonar_arff

    Returns:
        (X, y, n) -- (dataset, labels, #classes)
    """
    data, _ = loadarff(os.path.join("data/", "sonar_arff.arff"))
    X = [list(i)[:-1] for i in data]
    y = [i[-1] for i in data]
    return X, y, 2
