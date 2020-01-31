import numpy as np


def score(model):
    """Return the score according to AgnoS-W given the model.
    
    Arguments:
        model {Model} -- Trained AutoEncoder
    
    Returns:
        list -- Sorted features according to impact
    """    
    # get the first layer
    layer = model.get_layer('encoder')
    # extracts weights
    weights = layer.get_weights()[0]
    # calculate the infinity norm as shown in the paper. 
    # For each input feature get the absolute maximum weight
    # connected with this feature
    scores = np.linalg.norm(weights, ord=np.inf, axis=1)
    # the final score is a importance measure for each feature
    sorted_scores = sorted(range(len(scores)), key=lambda k: scores[k])
    return sorted_scores[::-1]
