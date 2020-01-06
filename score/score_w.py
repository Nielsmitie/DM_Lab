import numpy as np


def score(model):
    # get the first layer
    layer = model.get_layer('encoder')
    # extracts weights. Two weight matrices. First weights second bias
    weights = layer.get_weights()[0]
    # calculate the infinity norm as shown in the paper. For each input feature get the absolute maximum weight
    # connected with this feature
    score = np.linalg.norm(weights, ord=np.inf, axis=1)
    # the final score is a importance measure for each feature
    return score
