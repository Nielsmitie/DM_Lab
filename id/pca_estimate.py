from sklearn.decomposition import PCA


def get_id(x, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(x)
    return pca.n_components
