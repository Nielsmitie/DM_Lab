import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import warnings
import logging


def KATE(input_size, n_hidden, activation_hidden='tanh', activation_output='tanh', loss='binary_crossentropy', lr=1e-4,
         metrics=None, comp_topk=None, kernel_initializer='glorot_normal'):

    if metrics is None:
        metrics = ['mae']
    input_layer = Input(shape=input_size)

    encoded_layer = Dense(n_hidden, activation=activation_hidden, kernel_initializer=kernel_initializer,
                          name="encoder")(input_layer)

    encoded = KCompetitive(comp_topk, 'float')(encoded_layer)

    # "decoded" is the lossy reconstruction of the input
    # add non-negativity contraint to ensure probabilistic interpretations
    decoded = Dense_tied(input_size[0], activation=activation_output, tied_to=encoded_layer, name='decoder',
                         kernel_initializer=kernel_initializer)(encoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(outputs=decoded, inputs=input_layer)

    # this model maps an input to its encoded representation
    encoder = Model(outputs=encoded, inputs=input_layer)

    # create a placeholder for an encoded input
    encoded_input = Input(shape=(n_hidden,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    model = Model(outputs=decoder_layer(encoded_input), inputs=encoded_input)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr), loss=loss, metrics=metrics)

    return model


def get_model(*args, **kwargs):
    return KATE(*args, **kwargs)


class KCompetitive(Layer):
    '''Applies K-Competitive layer.
    # Arguments
    '''

    def __init__(self, topk, ctype, **kwargs):
        self.topk = topk
        self.ctype = ctype
        self.uses_learning_phase = True
        self.supports_masking = True
        super(KCompetitive, self).__init__(**kwargs)

    def call(self, x):
        if self.ctype == 'ksparse':
            return K.in_train_phase(self.kSparse(x, self.topk), x)
        elif self.ctype == 'kcomp':
            return K.in_train_phase(self.k_comp_tanh(x, self.topk), x)
        else:
            warnings.warn("Unknown ctype, using no competition.")
            return x

    def get_config(self):
        config = {'topk': self.topk, 'ctype': self.ctype}
        base_config = super(KCompetitive, self).get_config()
        base_config.update(config)
        return dict(base_config)

    def k_comp_tanh(self, x, topk, factor=6.26):
        logging.info('Run k_comp_tanh')
        dim = int(x.get_shape()[1])
        # batch_size = tf.to_float(tf.shape(x)[0])
        if topk > dim:
            warnings.warn('Warning: topk should not be larger than dim: %s, found: %s, using %s' % (dim, topk, dim))
            topk = dim

        P = (x + tf.abs(x)) / 2
        N = (x - tf.abs(x)) / 2

        values, indices = tf.nn.top_k(P,
                                      topk / 2)  # indices will be [[0, 1], [2, 1]], values will be [[6., 2.], [5., 4.]]
        # We need to create full indices like [[0, 0], [0, 1], [1, 2], [1, 1]]
        my_range = tf.expand_dims(tf.range(0, tf.shape(indices)[0]), 1)  # will be [[0], [1]]
        my_range_repeated = tf.tile(my_range, [1, topk / 2])  # will be [[0, 0], [1, 1]]
        full_indices = tf.stack([my_range_repeated, indices],
                                axis=2)  # change shapes to [N, k, 1] and [N, k, 1], to concatenate into [N, k, 2]
        full_indices = tf.reshape(full_indices, [-1, 2])
        P_reset = tf.sparse.to_dense(full_indices, tf.shape(x), tf.reshape(values, [-1]), default_value=0.,
                                     validate_indices=False)

        values2, indices2 = tf.nn.top_k(-N, topk - topk / 2)
        my_range = tf.expand_dims(tf.range(0, tf.shape(indices2)[0]), 1)
        my_range_repeated = tf.tile(my_range, [1, topk - topk / 2])
        full_indices2 = tf.stack([my_range_repeated, indices2], axis=2)
        full_indices2 = tf.reshape(full_indices2, [-1, 2])
        N_reset = tf.sparse.to_dense(full_indices2, tf.shape(x), tf.reshape(values2, [-1]), default_value=0.,
                                     validate_indices=False)

        # 1)
        # res = P_reset - N_reset
        # tmp = 1 * batch_size * tf.reduce_sum(x - res, 1, keep_dims=True) / topk

        # P_reset = tf.sparse_to_dense(full_indices, tf.shape(x),
        # tf.reshape(tf.add(values, tf.abs(tmp)), [-1]), default_value=0., validate_indices=False)
        # N_reset = tf.sparse_to_dense(full_indices2, tf.shape(x),
        # tf.reshape(tf.add(values2, tf.abs(tmp)), [-1]), default_value=0., validate_indices=False)

        # 2)
        # factor = 0.
        # factor = 2. / topk
        P_tmp = factor * tf.reduce_sum(P - P_reset, 1, keep_dims=True)  # 6.26
        N_tmp = factor * tf.reduce_sum(-N - N_reset, 1, keep_dims=True)
        P_reset = tf.sparse.to_dense(full_indices, tf.shape(x), tf.reshape(tf.add(values, P_tmp), [-1]),
                                     default_value=0., validate_indices=False)
        N_reset = tf.sparse.to_dense(full_indices2, tf.shape(x), tf.reshape(tf.add(values2, N_tmp), [-1]),
                                     default_value=0., validate_indices=False)

        res = P_reset - N_reset

        return res

    def kSparse(self, x, topk):
        logging.info('Run regular k-sparse')
        dim = int(x.get_shape()[1])
        if topk > dim:
            warnings.warn('Warning: topk should not be larger than dim: %s, found: %s, using %s' % (dim, topk, dim))
            topk = dim

        k = dim - topk
        values, indices = tf.nn.top_k(-x, k)  # indices will be [[0, 1], [2, 1]], values will be [[6., 2.], [5., 4.]]

        # We need to create full indices like [[0, 0], [0, 1], [1, 2], [1, 1]]
        my_range = tf.expand_dims(tf.range(0, tf.shape(indices)[0]), 1)  # will be [[0], [1]]
        my_range_repeated = tf.tile(my_range, [1, k])  # will be [[0, 0], [1, 1]]

        full_indices = tf.stack([my_range_repeated, indices],
                                axis=2)  # change shapes to [N, k, 1] and [N, k, 1], to concatenate into [N, k, 2]
        full_indices = tf.reshape(full_indices, [-1, 2])

        to_reset = tf.sparse.to_dense(full_indices, tf.shape(x), tf.reshape(values, [-1]), default_value=0.,
                                      validate_indices=False)

        res = tf.add(x, to_reset)

        return res

class Dense_tied(Dense):
    """
    A fully connected layer with tied weights.
    """
    def __init__(self, units,
                 activation=None, use_bias=True,
                 bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None,
                 tied_to=None, **kwargs):
        self.tied_to = tied_to

        super(Dense_tied, self).__init__(units=units,
                 activation=activation, use_bias=use_bias,
                 bias_initializer=bias_initializer,
                 kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                 activity_regularizer=activity_regularizer,
                 kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                 **kwargs)

    def build(self, input_shape):
        super(Dense_tied, self).build(input_shape)  # be sure you call this somewhere!
        if self.kernel in self.trainable_weights:
            self.trainable_weights.remove(self.kernel)


    def call(self, x, mask=None):
        # Use tied weights
        self.kernel = K.transpose(self.tied_to.kernel)
        output = K.dot(x, self.kernel)
        if self.use_bias:
            output += self.bias
        return self.activation(output)
