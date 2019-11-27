import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense


# hey used mean squared error for the ae. Page 3 Chapter 2.1
def AutoEncoder(input_size, n_hidden, activation_hidden='tanh', activation_output='tanh', loss='mean_squared_error',
                kernel_regularizer=None,
                activity_regularizer=None, kernel_constraint=None, lr=1e-4, metrics=None):
    if metrics is None:
        metrics = ['mae']
    input_layer = Input(shape=input_size)
    # todo maybe only input -> hidden has to have constraints
    # paper stated they used tanh activation, but should be changeable
    hidden = Dense(n_hidden, activation=activation_hidden, kernel_regularizer=kernel_regularizer,
                   activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint)(input_layer)
    # todo what activation function to use? Has to reflect all possible values of input. So can't be sigmoid or softmax
    # used None, tanh is bound to +/-1 and values can be more than +1 and less than -1 ?!
    output_layer = Dense(input_size[0], activation=activation_output)(hidden)
    autoencoder = Model(input_layer, output_layer)
    # TODO: should the loss module be implemented here or is it applied on a per layer basis? Maybe change.
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(lr=lr), loss=loss, metrics=metrics)
    return autoencoder


def get_model(*args, **kwargs):
    return AutoEncoder(*args, **kwargs)
