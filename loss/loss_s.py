import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.python.keras import backend as K
from tensorflow.keras.layers import Input, Dense, LocallyConnected1D, Reshape, Flatten
from tensorflow.keras.regularizers import l1
from tensorflow.keras.constraints import MinMaxNorm


def losses(alpha):
    # don't know if l1_l2 is the same as l2-l1 in the paper so reimplementing the procedure
    return {
        'activity_regularizer': None,
        'kernel_regularizer': None,
        'kernel_constraint': None,
        'slack_regularizer': add_slack_layer(alpha)
    }


def add_slack_layer(alpha):
    def wrapper(input_size, tmp_input_layer, kernel_initializer):
        with tf.name_scope('slack_layer'):
            # add another layer that multiplies the input feature with slack variable a
            # For this use a LocallyConnected1D Layer with filter-size 1
            # add l1 regularization that is added to the loss as shown in the paper
            tmp_input_layer = Reshape((input_size[0], 1), name='reshapeTo2D')(tmp_input_layer)
            tmp_input_layer = LocallyConnected1D(1, 1, strides=1,
                                                 kernel_initializer=kernel_initializer,
                                                 kernel_regularizer=l1(alpha),
                                                 # l1: The slack variables (weights of this layer) are added to loss
                                                 kernel_constraint=MinMaxNorm(min_value=0.0, max_value=1.0),
                                                 # constraint slack variable between 0 and 1
                                                 use_bias=False,  # only a_i * f_i is allowed
                                                 implementation=3,  # use sparse matrix multiply
                                                 name='slack_variables'
                                                 )(tmp_input_layer)
            tmp_input_layer = Flatten(name='flattenTo1D')(tmp_input_layer)
        return tmp_input_layer
    return wrapper
