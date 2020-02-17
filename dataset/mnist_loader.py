from tensorflow.keras.datasets import mnist
import numpy as np


def get_dataset():
    """Loads the MNIST dataset.

    Returns:
        (X, y, n) -- (dataset, labels, #classes)
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    X = np.append(x_train, x_test)
    y = np.append(y_train, y_test)
    X = X.reshape(70000, 784)
    return X, y, 10
