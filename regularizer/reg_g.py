import tensorflow as tf
from tensorflow.keras import losses as keras_loss
from tensorflow.python.keras import backend as K
from regularizer.reg_w import l1_minus_l2
from tensorflow.keras.backend import tanh


# TODO: WORK IN PROGRESS


def regularizer(alpha):
    """Return AgnoS-G gradient regularizer.

    Arguments:
        alpha {float} -- Regularization factor

    Returns:
        dict -- Dict containing gradient_regularizer
    """
    return {'gradient_regularizer': loss_wrapper(alpha)}


def loss_wrapper(alpha):
    """First layer to save alpha for the session.

    Arguments:
        alpha {float} -- Regularization factor

    Returns:
        function -- g_closure
    """
    def g_closure(intermediate_layer_model, loss_func):
        """Second layer. Wrapper around loss function, because loss can only take two arguments.

        Arguments:
            intermediate_layer_model {layer} -- Pointer to intermediate layer
            loss_func {function} -- Loss function

        Returns:
            function -- loss_g
        """
        loss_func = keras_loss.get(loss_func)

        def loss_g(y_true, y_pred):
            """Calculating the gradients of each hidden variable w.r.t. each original feature at every datapoint,
            then aggregating all these small gradients is both tough and very computationally expensive
            (three nested for loops) in the general case.

            Arguments:
                y_true {list} -- True labels
                y_pred {list} -- Predicted labels

            Returns:
                function -- loss
            """
            loss = loss_func(y_true, y_pred)

            #encoder_output = intermediate_layer_model.output
            #encoder_gradient = K.gradients(loss, [encoder_output])[0]
            #gradients = K.gradients(loss, intermediate_layer_model.inputs)
            #loss += alpha * tf.reduce_sum(tf.sqrt(gradients[0]))

            return loss
        return loss_g
    return g_closure


"""
def g_regularization(weights, dataset, model):
    reg = 0.
    for i in dataset.shape[1]:
        inner = 0.
        for x_k in dataset:
            for j in range(model.get_layer('encoder').output_shape[1]):
                inner += tf.square(tf.multiply(tf.expand_dims(weights['encoder'], 0), (1-tf.square(tanh(x_k[i])))))
        reg += tf.sqrt(inner)
    return reg
"""
