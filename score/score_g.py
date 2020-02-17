import numpy as np
import tensorflow as tf


# TODO: WORK IN PROGRESS
def score(model):
    """Gets the model and returns the scoring of the features.
    Only works if activation function is tanh due to
    tanh' =  1 - tanh².

    Arguments:
        model {Model} -- Trained AutoEncoder

    Returns:
        list -- Sorted features according to impact
    """
    # calculate the scores
    scores = tf.reduce_sum(tf.reduce_sum(tf.square(tf.multiply(tf.expand_dims(model.get_layer('encoder').get_weights()[0], 0),
                                                               (1 - tf.square(tf.expand_dims(model.get_layer('encoder'), 1))))), axis=1), axis=0)
    sorted_scores = sorted(range(len(scores)), key=lambda k: scores[k])
    return sorted_scores[::-1]
