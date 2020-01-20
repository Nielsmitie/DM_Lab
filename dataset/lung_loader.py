import pandas as pd
import numpy as np
import os


def get_dataset():
    """
    Load the lung cancer dataset.
    """
    lung = pd.read_csv(os.path.abspath("data/lung-cancer.data"), header=None)
    y = list(lung[1])
    lung = lung.drop(1, axis=1)
    lung.drop_duplicates(inplace=True)
    lung.replace('?', np.NaN, inplace=True)
    lung.dropna(inplace=True)
    X = lung.to_numpy()
    return X, y, len(set(y))
