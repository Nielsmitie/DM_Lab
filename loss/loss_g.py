import tensorflow as tf
from tensorflow.keras import losses as keras_loss
from tensorflow.python.keras import backend as K
from loss.loss_w import l1_minus_l2

# TODO: test if this is really corretly implemented


def losses(alpha):
    # don't know if l1_l2 is the same as l2-l1 in the paper so reimplementing the procedure
    return {
        'gradient_regularizer': loss_wrapper(alpha)}

# based on the build in regularizers from keras
# https://github.com/keras-team/keras/blob/master/keras/regularizers.py

#aggregate = tf.reduce_sum(tf.reduce_sum(tf.square(tf.multiply(tf.expand_dims(weights['encoder'],0),(1-tf.square(tf.expand_dims(encoder,1))))),axis=1),axis=0)))


def loss_wrapper(alpha):
    """
    First layer to save alpha for the session
    :param alpha: float value. Strength of regularization
    :return:
    """
    def g_closure(intermediate_layer_model, loss_func):
        """
        Second layer. Wrapper around loss function, because loss can take only two arguments.
        :param intermediate_layer_model: Pointer to
        :param loss_func:
        :return:
        """
        loss_func = keras_loss.get(loss_func)

        def loss_g(y_true, y_pred):
            loss = loss_func(y_true, y_pred)
            # encoder_output = intermediate_layer_model.output
            # encoder_gradient = K.gradients(loss, [encoder_output])[0]
            # paper: gradients of the encoder w.r.t an initial variable
            gradients = K.gradients(loss, intermediate_layer_model.inputs)

            loss += alpha * K.sum(tf.norm(gradients[0], ord=2))

            return loss
        return loss_g
    return g_closure
