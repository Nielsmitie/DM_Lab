import numpy as np
from skfeature.utility.unsupervised_evaluation import evaluation
from sklearn.neighbors import KNeighborsRegressor


def k_means_accuracy(x, y, num_clusters, feature_rank_values, top_n=100):
    """
    This function calculates ACC.

    Input
    -----
    x: {numpy array}, shape (n_samples, n_selected_features)
            input data on the selected features
    num_clusters: {int}
            number of clusters
    y: {numpy array}, shape (n_samples,)
            true labels
    feature_rank_values: {list}
            scores of the features
    top_n: {int or list}
            top n features to be selected

    Output
    ------
    results: {dict}
        Dictionary containing the ACC for every given top_n
    """
    if not isinstance(top_n, list):
        top_n = [top_n]

    ranking = np.argsort(feature_rank_values)
    results = {}
    for n in top_n:
        _, acc = evaluation(x[:, ranking[:n]], num_clusters, y)
        results[n] = acc

    return results


def r_squared(x, y, num_clusters, feature_rank_values, top_n=100):
    """
    This function calculates R².

    Input
    -----
    x: {numpy array}, shape (n_samples, n_selected_features)
            input data on the selected features
    num_clusters: {int}
            number of clusters
    y: {numpy array}, shape (n_samples,)
            true labels
    feature_rank_values: {list}
            scores of the features
    top_n: {int or list}
            top n features to be selected

    Output
    ------
    results: {dict}
        Dictionary containing the R² for every given top_n
    """
    if not isinstance(top_n, list):
        top_n = [top_n]

    ranking = np.argsort(feature_rank_values)
    results = {}
    for n in top_n:
        r_scores = []
        for feature in range(x.shape[1]):
            regressor = KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', p=2)
            target = x[:, feature]
            regressor.fit(x[:, ranking[:n]], target)
            r_scores.append(regressor.score(x[:, ranking[:n]], target))
        pred = np.mean(r_scores)
        results[n] = pred

    return results
