import math

from sklearn import metrics


def get_id(dataset):
    # calculate pairwise distance
    distances = metrics.pairwise_distances(dataset, Y=None, metric="euclidean", n_jobs=None)

    # two shortest distances for each point r1, r2
    shortest_distances = []
    for i in range(len(distances)):
        shortest_distances[i][0] = distances[i][0]
        shortest_distances[i][1] = distances[i][1]
        for j in range(len(distances[0])):
            if distances[i][j] < shortest_distances[i][0]:
                shortest_distances[i][1] = shortest_distances[i][0]
                shortest_distances[i][0] = distances[i][j]

    # compute µ = r2/r1
    r2r1_quotient = []
    for i in range(len(shortest_distances)):
        r2r1_quotient[i] = shortest_distances[i][1] / shortest_distances[i][0]

    # sort µ ascending
    r2r1_quotient.sort()

    # calculate f(µi) = i/N for each µ
    f = []
    for i in range(len(r2r1_quotient)):
        f[i] = i / len(r2r1_quotient)

    # calculate the sum of all - (log(1-f(µ)) / log(µ))
    d_sum = 0
    for i in range(len(f)):
        d_sum += - ((math.log(1 - f[i])) / math.log(r2r1_quotient[i]))

    return d_sum / len(f)

#TODO needs testing