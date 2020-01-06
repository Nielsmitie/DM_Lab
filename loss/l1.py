from tensorflow.keras.regularizers import l1


def losses(alpha):
    return {'activity_regularizer': None, 'kernel_regularizer': l1(alpha), 'kernel_constraint': None}
