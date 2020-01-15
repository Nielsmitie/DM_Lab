import numpy as np


def score(model):
    # from name_scope slack_layer, that is a composite of multiple layers get the locallyConnected1D layer
    layer = model.get_layer('slack_variables')
    # get the weights. Without bias, so only one weight matrix there
    weights = layer.get_weights()[0]
    # is a sparse matrix with only one non-zero entry per feature. Keras is extracting them automatically.
    score = np.abs(weights) # the final score is just the absolute value of each slack variable
    return score
