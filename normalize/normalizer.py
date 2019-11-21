from sklearn.preprocessing import StandardScaler


def normalize(x):
    return StandardScaler().fit_transform(x)
