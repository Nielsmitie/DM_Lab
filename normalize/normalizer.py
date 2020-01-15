from sklearn.preprocessing import StandardScaler


def normalize(x, with_mean=True):
    return StandardScaler(with_mean=bool(with_mean)).fit_transform(x)
