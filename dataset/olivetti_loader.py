from sklearn import datasets


def get_dataset(shuffle=True, random_state=None):
    """
    Load the Olivetti faces data-set from AT&T.
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html
    """
    olivetti = datasets.fetch_olivetti_faces(shuffle=shuffle, random_state=random_state)
    return olivetti.data, olivetti.target, 40
