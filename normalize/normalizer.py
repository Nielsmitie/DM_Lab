from sklearn.preprocessing import StandardScaler


def normalize(x, with_mean=True, with_std=True):
    """Standardize features by removing the mean and scaling to unit variance.

    Arguments:
        x {list} -- Dataset

    Keyword Arguments:
        with_mean {bool} -- If True, center the data before scaling. (default: {True})
        with_std {bool} -- If True, scale the data to unit variance. (default: {True})

    Returns:
        list -- Normalized dataset
    """
    return StandardScaler(with_mean=bool(with_mean),
                          with_std=bool(with_std)).fit_transform(x)
