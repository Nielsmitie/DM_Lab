import sys
import logging
import random
from skfeature.function.similarity_based import SPEC
from skfeature.function.similarity_based import lap_score
from skfeature.function.sparse_learning_based import MCFS
from skfeature.function.sparse_learning_based import NDFS
from skfeature.utility import construct_W
from skfeature.utility.sparse_learning import feature_ranking


def competitor(method, X, metric='cosine', weightMode='heatKernel', k=5, t=1, style=0,
               n_selected_features=1, n_clusters=5):
    """
    Input
    -----
    method: {string}
        selected competitor method (SPEC, LAP, MDFS, NDFS, RANDOM)
    X: {numpy array}, shape (n_samples, n_features)
        input data
    n_selected_features: {int}
        number of selected features to select (mcfs only)
    kwargs: {dictionary}
        style:  {int}
            style == -1, the first feature ranking function, use all eigenvalues
            style == 0, the second feature ranking function, use all except the 1st eigenvalue
            style >= 2, the third feature ranking function, use the first k except 1st eigenvalue
            (spec only)
        n_clusters: {int}
            number of clusters (default is 5; mcfs and ndfs only, normally equal to number of classes)
        metric: {string}
            choices for different distance measures
            'euclidean' - use euclidean distance
            'cosine' - use cosine distance (default)
        weight_mode: {string}
            indicates how to assign weights for each edge in the graph
            'binary' - 0-1 weighting, every edge receives weight of 1 (default)
            'heat_kernel' - if nodes i and j are connected, put weight W_ij = exp(-norm(x_i - x_j)/2t^2)
                            this weight mode can only be used under 'euclidean' metric and you are required
                            to provide the parameter t
            'cosine' - if nodes i and j are connected, put weight cosine(x_i,x_j).
                        this weight mode can only be used under 'cosine' metric
        k: {int}
            choices for the number of neighbors (default k = 5)
        t: {float}
            parameter for the 'heat_kernel' weight_mode
            
    Output
    ------
    Returns either the score or W depending on the selected competitor method.
    score: {numpy array}, shape(n_features,)
        score of the similarity_based methods for each feature
    """
    num_features = X.shape[1]
    if n_selected_features > num_features:
        n_selected_features = num_features
        logging.warn("More features to select than given!")

    if method.upper() == "RANDOM":
        return random.choices(list(range(num_features)), k=n_selected_features)

    kwargs = {"metric": metric, "neighborMode": "knn", "weightMode": weightMode, "k": k, 't': t}
    W = construct_W.construct_W(X, **kwargs)  # affinity matrix

    if method.upper() == "SPEC":
        return SPEC.feature_ranking(SPEC.spec(X, **{'style': style}, W=W), **{'style': style})
    elif method.upper() == "LAP":
        return lap_score.feature_ranking(lap_score.lap_score(X, W=W))
    elif method.upper() == "MCFS":
        return MCFS.feature_ranking(MCFS.mcfs(X, n_selected_features=n_selected_features, W=W, n_clusters=n_clusters))
    elif method.upper() == "NDFS":
        return feature_ranking(NDFS.ndfs(X, W=W, n_clusters=n_clusters))
    else:
        logging.error("Method not known!")


def get_model(*args, **kwargs):
    return competitor(*args, **kwargs)
