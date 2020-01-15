import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer


def get_dataset():
    census = pd.read_csv(os.path.abspath("data/adult.data"), header=None)
    labels = census[14]
    census = census.drop(14, axis=1)
    vectors = None
    return np.array(vectors), list(labels), 2
