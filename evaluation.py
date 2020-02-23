import numpy as np
from skfeature.utility.unsupervised_evaluation import evaluation
from sklearn.neighbors import KNeighborsRegressor
import multiprocessing as mp


# TODO: Also return NMI
def score_acc(n_, X_, sorted_features_, num_clusters_, y_, repetitions_):
    nmi_total = 0
    acc_total = 0
    for _ in range(repetitions_):
        nmi, acc = evaluation(X_[:, sorted_features_[:n_]], num_clusters_, y_)
        nmi_total += nmi
        acc_total += acc
    return n_, float(acc_total) / repetitions_


def k_means_accuracy(X, y, num_clusters, sorted_features,
                     top_n=100, repetitions=20):
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

    pool = mp.Pool(mp.cpu_count())

    results = dict(pool.starmap(score_acc, [(n, X, sorted_features, num_clusters, y, repetitions) for n in top_n]))

    return results


def score_r_square(n_, X_, sorted_features_):
    r_scores = []
    for feature in range(X_.shape[1]):
        regressor = KNeighborsRegressor(
            n_neighbors=5, weights='uniform', algorithm='auto', p=2)
        target = X_[:, feature]
        regressor.fit(X_[:, sorted_features_[:n_]], target)
        r_scores.append(regressor.score(X_[:, sorted_features_[:n_]], target))
    pred = np.mean(r_scores)
    return n_, pred


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

    pool = mp.Pool(mp.cpu_count())

    results = dict(pool.starmap(score_r_square, [(n, X, sorted_features) for n in top_n]))

    print('r_square evaluation done. ({})'.format(results))

    return results
