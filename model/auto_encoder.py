import tensorflow as tf
from tensorflow.keras import layers, Model


def AutoEncoder(input_size, n_hidden, activation, loss='sparse_categorical_crossentropy', kernel_regularizer=None,
                activity_regularizer=None, kernel_constraint=None, lr=1e-4, metrics=None):
    if metrics is None:
        metrics = ['accuracy']
    input_layer = layers.Input(input_size)
    hidden = layers.Dense(n_hidden, activation=activation, kernel_regularizer=kernel_regularizer,
                          activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint)(input_layer)

    output_layer = layers.Dense(input_size, activation=activation, kernel_regularizer=kernel_regularizer,
                                activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint)(hidden)

    learner = Model(input_layer, output_layer)

    # TODO: should the loss module be implemented here or is it applied on a per layer basis? Maybe change.
    learner.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
                    loss=loss,
                    metrics=metrics)

    return learner


def get_model(*args, **kwargs):
    return AutoEncoder(*args, **kwargs)
