import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense


def AutoEncoder(input_size, n_hidden, activation, loss='binary_crossentropy', kernel_regularizer=None,
                activity_regularizer=None, kernel_constraint=None, lr=1e-4, metrics=['accuracy']):
    input_layer = Input(shape=input_size)
    hidden = Dense(n_hidden, activation=activation, kernel_regularizer=kernel_regularizer,
                          activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint)(input_layer)
    output_layer = Dense(input_size[0], activation=activation, kernel_regularizer=kernel_regularizer,
                          activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint)(hidden)
    autoencoder = Model(input_layer, output_layer)
    # TODO: should the loss module be implemented here or is it applied on a per layer basis? Maybe change.
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(lr=lr), loss='binary_crossentropy', metrics=metrics)
    return autoencoder


def get_model(*args, **kwargs):
    return AutoEncoder(*args, **kwargs)
