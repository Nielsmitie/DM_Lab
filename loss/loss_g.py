import tensorflow as tf
from tensorflow.keras import losses as keras_loss
from tensorflow.python.keras import backend as K
from loss.loss_w import l1_minus_l2

# todo test if this is really corretly implemented


def losses(alpha):
    # don't know if l1_l2 is the same as l2-l1 in the paper so reimplementing the procedure
    return {
        'gradient_regularizer': loss_wrapper(alpha)}

# based on the build in regularizers from keras
# https://github.com/keras-team/keras/blob/master/keras/regularizers.py

'''
def g_closure(intermediate_layer_model, loss):
    intermediate_output = intermediate_layer_model.predict(data)
    gradient = K.gradients(loss, intermediate_output.output)
'''


def loss_wrapper(alpha):
    def g_closure(intermediate_layer_model, loss_func):
        loss_func = keras_loss.get(loss_func)

        def loss_g(y_true, y_pred):
            loss = loss_func(y_true, y_pred)

            gradients = K.gradients(loss, intermediate_layer_model.inputs)

            loss += alpha * K.sum(tf.norm(gradients, ord=2))
        return loss_g
    return g_closure