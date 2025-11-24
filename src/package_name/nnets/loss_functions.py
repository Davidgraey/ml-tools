import numpy as np

loss_dictionary = {}

loss_func = lambda f: loss_dictionary.setdefault(f.__name__, f)

derivative_dictionary = {}

derivative = lambda f: derivative_dictionary.setdefault(f.__name__, f)


@loss_func
def rmse(prediction, T, pointwise = True):
    '''
    ROOT MEAN SQUARED ERRORS FOR REGRESSION MODELS
    :param prediction: numpy array of predictions (outputs)
    :param T: True or T values - dimensionally match prediction
    :return: evaluation of error / loss
    '''
    return np.expand_dims(np.sqrt(np.mean((T - prediction) ** 2)), -1)

@loss_func
def mse(prediction, T, pointwise = True):
    '''
    MEAN SQUARED ERRORS (L2) FOR REGRESSION MODELS
    :param prediction: numpy array of predictions (outputs)
    :param T: True or T values - dimensionally match prediction
    :return: evaluation of error / loss
    '''
    loss = np.expand_dims(0.5 * np.mean((prediction - T) ** 2), -1)
    return loss

#------------------------------------------------------------------
#DERIVATIVES

@derivative
def rmse_derivative(prediction, T):
    return np.abs(T - prediction) / np.sqrt(prediction.shape[0])

@derivative
def mse_derivative(prediction, T):
    return prediction - T