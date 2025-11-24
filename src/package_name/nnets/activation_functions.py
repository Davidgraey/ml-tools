import numpy as np
import jax.numpy as jnp
from numba import njit
import copy

EPSILON = 1e-12

# For numpy - install SVML
# conda install -c numba icc_rt
# @njit(cache=True,fastmath=True, parallel=True)

activation_dictionary = {}
activation = lambda f: activation_dictionary.setdefault(f.__name__, f)

# TODO: replace the activation dictionary setdefault with enums
# TODO: expand acitvation functions
# TODO: add GELU
# TODO: add swish
# TODO: add MISH - x * tanh(softplus(x))
# TODO: implement in jax
# import jax.numpy as jnp


@activation
def linear(x):
    '''
    **********ARGUMENTS**********
    :param x: incoming values in numpy array
    **********RETURNS**********
    :return: returns x
    '''
    return x


@activation
def sigmoid(x):
    '''
    Returns results of Sigmoid activation function
    **********ARGUMENTS**********
    :param x: incoming values in numpy array
    **********RETURNS**********
    :return: evaluation (single val for input_sample (row) of x)
    '''
    return 1 / (1 + np.exp(-x))


@activation
def tanh(x):
    '''
    **********ARGUMENTS**********
    :param x: incoming values in numpy array
    **********RETURNS**********
    :return: evaluation (single val for input_sample (row) of x)
    '''
    # could be done with np.tanh(x) as well
    # return 2 / (1 + np.e ** (-2 * x)) - 1
    return np.tanh(x)


@activation
def relu(x):
    '''**********ARGUMENTS**********
    :param x: incoming values in numpy array
    **********RETURNS**********
    :return: evaluation (single val for input_sample (row) of x)
    '''
    return np.maximum(0, x)


@activation
def relu_leaky(x, alpha = 0.1):
    '''**********ARGUMENTS**********
    :param x: incoming values in numpy array
    **********RETURNS**********
    :return: evaluation (single val for input_sample (row) of x)
    '''
    return np.where(x > 0, x, alpha * x)


@activation
def softmax(x):
    '''N-dimensional vector with values that sum to one - probabilistic multiclass
    **********ARGUMENTS**********
    :param x: incoming values in numpy array
    **********RETURNS**********
    :return: evaluation (single val for input_sample (row) of x)
    '''
    shifted = x - np.max(x)
    exps = np.exp(shifted) #shifted
    final = exps / (np.sum(exps, axis=1, keepdims=True) + EPSILON) #axis=1?
    return final

derivative_dictionary = {}

derivative = lambda f: derivative_dictionary.setdefault(f.__name__[:-11], f)


@derivative
def sigmoid_derivative(x):
    '''**********ARGUMENTS**********
    :param x: incoming values in numpy array
    **********RETURNS**********
    :return: evaluation (single val for input_sample (row) of x)
    '''
    return x * (1 - x)


@derivative
def relu_derivative(x):
    '''**********ARGUMENTS**********
    :param x: incoming values in numpy array
    **********RETURNS**********
    :return: evaluation (single val for input_sample (row) of x)
    '''
    der_x = copy.copy(x)
    return np.where(der_x < 0, 0, 1)


@derivative
def relu_leaky_derivative(x, alpha = 0.1):
    '''**********ARGUMENTS**********
    :param x: incoming values in numpy array
    **********RETURNS**********
    :return: evaluation (single val for val of x)
    '''
    der_x = np.ones_like(x)
    return np.where(der_x < 0, alpha, 1)


@derivative
def tanh_derivative(x):
    '''**********ARGUMENTS**********
    :param x: incoming values in numpy array
    **********RETURNS**********
    :return: evaluation (single val for input_sample (row) of x)
    '''
    return 1 - x ** 2


@derivative
def linear_derivative(x):
    '''**********ARGUMENTS**********
    :param x: incoming values in numpy array
    **********RETURNS**********
    :return: value of 1 (1st derivative of linear func x = 1)
    '''
    return np.ones_like(x)  # x


@derivative
def softmax_derivative(x):
    '''**********ARGUMENTS**********
    :param x: incoming values in numpy array
    **********RETURNS**********
    :return: evaluation
    '''
    #s = 1-x * x
    J = - x[..., None] * x[:, None, :]  # off-diagonal Jacobian
    iy, ix = np.diag_indices_from(J[0])
    J[:, iy, ix] = x * (1. - x)  # diagonal
    summed = np.sum(J, axis = 1)  # sum across rows
    return summed
