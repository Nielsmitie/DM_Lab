import numpy as np


def score(model):
    """Return the score according to AgnoS-S given the model.

    Arguments:
        model {Model} -- Trained AutoEncoder

    Returns:
        list -- Sorted features according to impact
    """
    # from name_scope slack_layer, that is a composite of multiple layers get
    # the locallyConnected1D layer
    layer = model.get_layer('slack_variables')
    # get the weights. Without bias, so only one weight matrix there
    weights = layer.get_weights()[0]
    # is a sparse matrix with only one non-zero entry per feature. Keras is
    # extracting them automatically.
    # the final score is just the absolute value of each slack variable
    scores = np.abs(weights)
    sorted_scores = sorted(range(len(scores)), key=lambda k: scores[k])
    return sorted_scores[::-1]
