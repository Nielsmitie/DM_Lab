import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, LocallyConnected1D, Reshape, Flatten
from tensorflow.keras.constraints import UnitNorm


# they used mean squared error for the ae. Page 3 Chapter 2.1
def AutoEncoder(input_size, n_hidden, activation_hidden='tanh', activation_output='tanh', loss='mean_squared_error',
                kernel_regularizer=None, gradient_regularizer=None, slack_regularizer=None,
                activity_regularizer=None, kernel_constraint=None, lr=1e-4, metrics=None,
                kernel_initializer='glorot_normal'):
    if metrics is None:
        metrics = ['mae']
    input_layer = Input(shape=input_size)
    # for the integration of slack_regularizer, share an object with hidden dense layer
    tmp_input_layer = input_layer
    # now input_layer can stay intact as an input Layer

    if slack_regularizer:
        tmp_input_layer = slack_regularizer(input_size, tmp_input_layer, kernel_initializer)

    # paper stated they used tanh activation, but should be changeable
    hidden = Dense(n_hidden, activation=activation_hidden, kernel_regularizer=kernel_regularizer,
                   activity_regularizer=activity_regularizer,
                   kernel_initializer=kernel_initializer, kernel_constraint=UnitNorm(axis=[0, 1]),
                   name='encoder')(tmp_input_layer)
    # TODO: what glorot method is used glorot_normal or glorot_uniform. Both are mentioned in the same paper.
    # used None, tanh is bound to +/-1 and values can be more than +1 and less than -1 ?!
    # should this be also constraint? No, the paper says only the encoder is constraint
    output_layer = Dense(input_size[0], activation=activation_output,
                         kernel_initializer=kernel_initializer,
                         name='decoder')(hidden)
    autoencoder = Model(input_layer, output_layer)

    if gradient_regularizer:
        layer_name = 'encoder'

        intermediate_layer_model = Model(inputs=autoencoder.input,
                                         outputs=autoencoder.get_layer(layer_name).output)

        loss = gradient_regularizer(intermediate_layer_model, loss)

    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(lr=lr), loss=loss, metrics=metrics)
    return autoencoder


def get_model(*args, **kwargs):
    return AutoEncoder(*args, **kwargs)
