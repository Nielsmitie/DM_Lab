from sklearn.cluster import KMeans
import numpy as np
from sklearn import metrics
# todo k-means clustering accuracy


def k_means_accuracy(x, y, num_clusters, feature_rank_values, top_n=100):
    # todo is this the "accuracy" used in the papaer
    # https://stackoverflow.com/questions/34047540/python-clustering-purity-metric
    def purity_score(y_true, y_pred):
        # compute contingency matrix (also called confusion matrix)
        contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
        # return purity
        return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

    kmeans = KMeans(n_clusters=num_clusters, n_jobs=-1)

    if not isinstance(top_n, list):
        top_n = [top_n]

    # sort the values
    ranking = np.argsort(feature_rank_values)

    results = {}
    for n in top_n:
        # take the top_n for training
        pred = kmeans.fit_predict(x[:, ranking[:n]], y)

        results[n] = purity_score(y, pred)
        print('Purity score: ' + str(results[n]))

    return results


