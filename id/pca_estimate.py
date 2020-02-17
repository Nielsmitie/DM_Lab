from sklearn.decomposition import PCA


def get_id(dataset, n_components):
    """Estimate Intristic Dimension using PCA.

    Arguments:
        dataset {list} -- Dataset
        n_components {int, float, None or str} -- Number of components to keep.

    Returns:
        int -- Intristic Dimension
    """
    pca = PCA(n_components=n_components)
    tmp = pca.fit_transform(dataset)
    return tmp.shape[1]
