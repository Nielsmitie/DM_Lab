import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.python.keras import backend as K

# todo test if this is really corretly implemented
# observation: The loss is decreasing faster and with less deviation. Maybe even further
# compare without regularization: 3.12.19 15:35:02
# with regularization: 3.12.19 15:43:36
# but both versions reach the same purity score


def losses(alpha):
    # don't know if l1_l2 is the same as l2-l1 in the paper so reimplementing the procedure
    return {
        'activity_regularizer': l1_minus_l2(1.0),
        'kernel_regularizer': None,
        'kernel_constraint': None}

# based on the build in regularizers from keras
# https://github.com/keras-team/keras/blob/master/keras/regularizers.py


class L2MinusL1(regularizers.Regularizer):

    def __init__(self, alpha=1.0):
        self.alpha = K.cast_to_floatx(alpha)

    def __call__(self, x):
        norm = tf.norm(x, ord=2)
        regularization = self.alpha * K.sum(norm)

        return regularization

    def get_config(self):
        return {'l1_minus_l2': self.alpha}


def l1_minus_l2(alpha=1.0):
    return L2MinusL1(alpha=alpha)


class L1L2(regularizers.Regularizer):
    """Regularizer for L1 and L2 regularization.
    # Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
    """

    def __init__(self, l1=0., l2=0.):
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)

    def __call__(self, x):
        regularization = 0.
        if self.l1:
            test1 = K.abs(x)
            tmp1 = K.sum(test1)
            regularization += self.l1 * tmp1
        if self.l2:
            test2 = K.square(x)
            tmp2 = K.sum(test2)
            regularization += self.l2 * tmp2
        return regularization

    def get_config(self):
        return {'l1': float(self.l1),
                'l2': float(self.l2)}


# Aliases.


def l1(l=0.01):
    return L1L2(l1=l)


def l2(l=0.01):
    return L1L2(l2=l)


def l1_l2(l1=0.01, l2=0.01):
    return L1L2(l1=l1, l2=l2)

