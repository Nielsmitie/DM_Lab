from tensorflow.keras import regularizers
from tensorflow.python.keras import backend as K


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
        norm = K.norm(x, ord=2, axis=1)
        regularization = K.sum(norm)

        return regularization

    def get_config(self):
        return {'l1_minus_l2': self.alpha}


def l1_minus_l2(alpha=1.0):
    return L2MinusL1(alpha=alpha)
