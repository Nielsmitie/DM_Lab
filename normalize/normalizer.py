from sklearn.preprocessing import StandardScaler


def normalize(x, with_mean=True):
    """
    Standardize features by removing the mean and scaling to unit variance.
    """
    return StandardScaler(with_mean=bool(with_mean)).fit_transform(x)
