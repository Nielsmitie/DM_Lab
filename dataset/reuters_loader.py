from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer


# TODO!
def get_dataset():
    data, target = datasets.fetch_rcv1(return_X_y=True)
    return vectors, target, 103
