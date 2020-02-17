import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l1_l2
from tensorflow.python.keras import backend as K


# TODO: Compare to L1-L2 from Keras
def regularizer(alpha):
    """Return kernel regularizer using L1-L2.

    Arguments:
        alpha {float} -- Regularization factor

    Returns:
        dict -- Dict containing kernel_regularizer with L1-L2 regularizer
    """
    return {'kernel_regularizer': l1_minus_l2(alpha)}


class L2MinusL1(regularizers.Regularizer):
    """
    Wrapper to save the strength of the regularization outside of the training loop.
    """

    def __init__(self, alpha=1.0):
        self.alpha = K.cast_to_floatx(alpha)

    def __call__(self, x):
        """
        Implementation from the paper.
        """
        norm = tf.norm(x, ord=2, axis=1)
        regularization = self.alpha * K.sum(norm)

        return regularization

    def get_config(self):
        return {'l1_minus_l2': self.alpha}


def l1_minus_l2(alpha=1.0):
    """Wrapper for L2MinusL1"""
    return L2MinusL1(alpha=alpha)
