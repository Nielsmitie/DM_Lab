

def regularizer(alpha=0):
    """Returns a dummy-regularizer.
    
    Returns:
        dict -- Dict containing only None-values
    """
    return {'activity_regularizer': None,
            'bias_regularizer': None,
            'kernel_regularizer': None}
