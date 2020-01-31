import sys
import logging
import random
from skfeature.function.similarity_based import SPEC
from skfeature.function.similarity_based import lap_score
from skfeature.function.sparse_learning_based import MCFS
from skfeature.function.sparse_learning_based import NDFS
from skfeature.utility import construct_W
from skfeature.utility.sparse_learning import feature_ranking


def competitor(method, X, metric='cosine', weight_mode='heat_kernel', k=5, t=1, style=0,
               n_selected_features=None, n_clusters=5):
    """Executes competitor FS methods like RANDOM, SPEC, LAP, MDFS or NDFS.
    
    Arguments:
        method {str} -- Name of the Method (RANDOM, SPEC, LAP, MDFS, NDFS)
        X {list} -- Dataset
    
    Keyword Arguments:
        metric {str} -- Choices for different distance measures (default: {'cosine'})
            'euclidean' - use euclidean distance
            'cosine' - use cosine distance
        weight_mode {str} -- Indicates how to assign weights for each edge in the graph (default: {'heat_kernel'})
            'binary' - 0-1 weighting, every edge receives weight of 1 (default)
            'heat_kernel' - if nodes i and j are connected, put weight W_ij = exp(-norm(x_i - x_j)/2t^2)
                            this weight mode can only be used under 'euclidean' metric and you are required
                            to provide the parameter t
            'cosine' - if nodes i and j are connected, put weight cosine(x_i,x_j).
                        this weight mode can only be used under 'cosine' metric
        k {int} -- Choices for the number of neighbors (default: {5})
        t {int} -- Parameter for the 'heat_kernel' weight_mode (default: {1})
        style {int} -- Style parameter for SPEC (default: {0})
            style == -1, the first feature ranking function, use all eigenvalues
            style == 0, the second feature ranking function, use all except the 1st eigenvalue
            style >= 2, the third feature ranking function, use the first k except 1st eigenvalue
        n_selected_features {None or int} -- Number of selected features (default: {None})
        n_clusters {int} -- Number of clusters; should be equal to number of classes (default: {5})
    
    Returns:
        list -- Sorted list of features, starting with the highest score.
    """
    if n_selected_features is None:
        n_selected_features = X.shape[1] # Select all features
    num_features = X.shape[1]
    if n_selected_features > num_features:
        n_selected_features = num_features
        logging.warn("More features to select than given!")

    if method.upper() == "RANDOM":
        ranking = list(range(X.shape[1]))
        random.shuffle(ranking)
        return ranking

    # affinity matrix
    kwargs = {"metric": metric, "neighborMode": "knn", "weight_mode": weight_mode, "k": k, 't': t}
    W = construct_W.construct_W(X, **kwargs)

    if method.upper() == "SPEC":
        return SPEC.feature_ranking(SPEC.spec(X, **{'style': style}), **{'style': style})
    elif method.upper() == "LAP":
        return lap_score.feature_ranking(lap_score.lap_score(X, W=W))
    elif method.upper() == "MCFS":
        return MCFS.feature_ranking(MCFS.mcfs(X, n_selected_features=n_selected_features, W=W, n_clusters=n_clusters))
    elif method.upper() == "NDFS":
        return feature_ranking(NDFS.ndfs(X, W=W, n_clusters=n_clusters))
    else:
        logging.error("Method not known!")


def get_model(*args, **kwargs):
    """Returns result of competitor method.
    
    Returns:
        list -- Scores of the features
    """    
    return competitor(*args, **kwargs)
