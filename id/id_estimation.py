import math
import numpy as np

from sklearn import metrics

def get_id(dataset):
    # calculate pairwise distance
    distances = metrics.pairwise_distances(dataset, Y=None, metrics="euclidian", n_jobs=None)

    # two shortest distances for each point r1, r2
    shortest_distances = []
    shortest_distances = np.argsort(distances, axis=0)

    # compute µ = r2/r1
    r2r1_quotient = []
    for i in range(len(shortest_distances)):
        r2r1_quotient.append(distances[shortest_distances[i][2]] / distances[shortest_distances[i][1]])

    # sort µ ascending
    r2r1_ind = np.argsort(r2r1_quotient)

    # calculate f(µi) = i/N for each µ
    f = []
    for i in range(len(r2r1_ind)):
        f.append(i / len(r2r1_ind))

    # calculate the sum of all - (log(1-f(µ)) / log(µ))
    d_sum = 0
    for i in range(len(f)):
        if r2r1_quotient[i][r2r1_ind[i][1]] == 1:
            #r2r1_quotient[i][r2r1_ind[i][1]] = 0.99
            continue
        print(r2r1_quotient[i][r2r1_ind[i][1]])
        d_sum +=  ((math.log(1 - f[i])) / math.log(r2r1_quotient[i][r2r1_ind[i][1]]))

    return d_sum / len(f)

#TODO needs testing