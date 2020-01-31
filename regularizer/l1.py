from tensorflow.keras.regularizers import l1


def regularizer(alpha):
    """Return L1 regularization.

    Arguments:
        alpha {float} -- Regularization factor

    Returns:
        dict -- Dict containing regularizers
    """
    return {'kernel_regularizer': l1(alpha)}
