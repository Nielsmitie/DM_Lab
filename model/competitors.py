import sys
from skfeature.function.similarity_based.SPEC import spec
from skfeature.function.similarity_based.lap_score import lap_score
from skfeature.function.sparse_learning_based.MCFS import mcfs
from skfeature.function.sparse_learning_based.NDFS import ndfs
from skfeature.utility import construct_W

"""
Input
----
method: {string}
    selected competitor method (SPEC, LAP, MDFS, NDFS)
X:      {numpy array}, shape (n_samples, n_features)
    input data
n_selected_features:    {int}
    number of selected features to select (mcfs only)
kwargs: {dictionary}
    style:  {int}
        style == -1, the first feature ranking function, use all eigenvalues
        style == 0, the second feature ranking function, use all except the 1st eigenvalue
        style >= 2, the third feature ranking function, use the first k except 1st eigenvalue
        (spec only)
    n_clusters: {int}
        number of clusters (default is 5; mcfs and ndfs only)
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
----
Returns either the score or W depending on the selected competitor method.
score:  {numpy array}, shape(n_features,)
    score of the similarity_based methods (lap and spec) for each feature
    
W:      {numpy array}, shape(n_features, n_clusters)
    feature weight matrix of the sparse_learning_based methods (mcfs, ndfs) 
"""


def competitor(method, X, metric='cosine', weightMode='binary', k=5, t=0, style=0,
               n_selected_features=20, n_clusters=5):
    kwargs = {"metric": metric, "weightMode": weightMode, "k": k, 't': t}
    W = construct_W.construct_W(X, **kwargs)  # affinity matrix

    if method == "SPEC":
        return spec(X, **{'style': style}, W=W)
    elif method == "LAP":
        return lap_score(X, W=W)
    elif method == "MCFS":
        return mcfs(X, n_selected_features, W=W, n_clusters=n_clusters)
    elif method == "NDFS":
        return ndfs(X, W=W, n_clusters=n_clusters)
    else:
        print("Method not known!", file=sys.stderr)


def get_competitor(*args, **kwargs):
    return competitor(*args, **kwargs)
