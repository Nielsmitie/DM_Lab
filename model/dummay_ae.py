import tensorflow as tf
from tensorflow.keras import layers, Model


def AutoEncoder(input_size, n_hidden, activation):
    input_layer = layers.Input(input_size)
    hidden = layers.Dense(512, activation=activation)(input_layer)

    output_layer = layers.Dense(input_size, activation=activation)(hidden)

    return Model(input_layer, output_layer)


def get_model(input_size, n_hidden, activation):
    return AutoEncoder(input_size, n_hidden, activation)
