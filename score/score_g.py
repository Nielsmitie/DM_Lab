import numpy as np


# TODO!
def score(model):
    # get the first layer
    layer = model.get_layer('encoder')
    # extracts weights
    weights = layer.get_weights()[0]
    # calculate the scores
    grad = 0
    scores = np.amax(weights*(1-grad**2), axis=1)
    # the final score is a importance measure for each feature
    return scores
