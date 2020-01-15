from sklearn.decomposition import PCA


def get_id(x, n_components):
    """
    Estimate Intristic Dimension using PCA.
    """
    pca = PCA(n_components=n_components)
    tmp = pca.fit_transform(x)
    return tmp.shape[1]
