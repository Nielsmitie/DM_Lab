import numpy as np
from skfeature.utility.unsupervised_evaluation import evaluation
from sklearn.neighbors import KNeighborsRegressor


def k_means_accuracy(X, y, num_clusters, sorted_features, top_n=100, repetitions=20):
    """This function calculates the ACC.
    
    Arguments:
        X {list} -- input data on the selected features (n_samples, n_selected_features)
        y {list} -- true labels (n_samples,)
        num_clusters {int} -- Number of clusters
        sorted_features {list} -- Features sorted by their impact
        repetitions {int} -- Number of times to repeat clustering
    
    Keyword Arguments:
        top_n {int or list} -- Top n features to be selected (default: {100})
    
    Returns:
        dict -- Dictionary containing the ACC for every given top_n
    """
    if not isinstance(top_n, list):
        top_n = [top_n]

    # TODO: Also return NMI
    results = {}
    for n in top_n:
        nmi_total = 0
        acc_total = 0
        for _ in range(repetitions):
            nmi, acc = evaluation(X[:, sorted_features[:n]], num_clusters, y)
            nmi_total += nmi
            acc_total += acc
        results[n] = float(acc)/repetitions

    return results


def r_squared(X, y, num_clusters, sorted_features, top_n=100):
    """This function calculates the R² score (aka coefficient of determination).
    R²(f, S) is in (-inf, 1]
    
    Arguments:
        X {list} -- input data on the selected features (n_samples, n_selected_features)
        y {list} -- true labels (n_samples,)
        num_clusters {int} -- Number of clusters
        sorted_features {list} -- Features sorted by their impact
    
    Keyword Arguments:
        top_n {int or list} -- Top n features to be selected (default: {100})
    
    Returns:
        dict -- Dictionary containing the R² for every given top_n
    """
    if not isinstance(top_n, list):
        top_n = [top_n]

    results = {}
    for n in top_n:
        r_scores = []
        for feature in range(X.shape[1]):
            regressor = KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', p=2)
            target = X[:, feature]
            regressor.fit(X[:, sorted_features[:n]], target)
            r_scores.append(regressor.score(X[:, sorted_features[:n]], target))
        pred = np.mean(r_scores)
        results[n] = pred

    return results
