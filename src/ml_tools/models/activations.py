import numpy as np
from numpy.typing import NDArray

import copy
from typing import Optional

EPSILON = 1e-15

activation_dictionary = {}
activation = lambda f: activation_dictionary.setdefault(f.__name__, f)
derivative_dictionary = {}

derivative = lambda f: derivative_dictionary.setdefault(f.__name__[:-11], f)


@activation
def sigmoid(x_array: NDArray) -> NDArray:
    """Numerically stable version of sigmoid"""
    result = np.empty_like(x_array)

    # Handle positive x
    positive_mask = x_array >= 0
    result[positive_mask] = 1 / (1 + np.exp(-x_array[positive_mask]))

    # Handle negative x
    negative_mask = x_array < 0
    exp_x = np.exp(x_array[negative_mask])
    result[negative_mask] = exp_x / (1 + exp_x)

    return result


# def _sig_pos(x):
#     return 1 / (1 + np.exp(-x))
#
#
# def _sig_neg(x):
#     return np.exp(x) / (1 + np.exp(x))
#
# @activation
# def sigmoid(x):
#     '''
#     Returns results of Sigmoid activation function
#     **********ARGUMENTS**********
#     :param x: incoming values in numpy array
#     **********RETURNS**********
#     :return: evaluation (single val for input_sample (row) of x)
#     '''
#
#     return np.piecewise(x, [x > 0], [_sig_pos, _sig_neg])


@activation
def swish(x: NDArray):
    """
    f(x) = x * σ(x)
    Parameters
    ----------
    x :

    Returns
    -------

    """

    return x * sigmoid(x)


@activation
def softmax(x_array: NDArray) -> NDArray:
    """
    N-dimensional vector with values that sum to one - probabilistic multiclass
    Parameters
    ----------
    x_array : incoming values in numpy array

    Returns
    -------
    evaluation (single val for input_sample (row) of x)
    """

    exps = np.exp(x_array - np.max(x_array, axis=-1, keepdims=True))
    return exps / np.sum(exps, axis=-1, keepdims=True)


@activation
def linear(x_array: NDArray) -> NDArray:
    """linear forward - for regression"""
    return x_array


@activation
def tanh(x: NDArray) -> NDArray:
    """
    **********ARGUMENTS**********
    :param x: incoming values in numpy array
    **********RETURNS**********
    :return: evaluation (single val for input_sample (row) of x)
    """
    # could be done with np.tanh(x) as well
    return 2 / (1 + np.e ** (-2 * x)) - 1


@activation
def relu(x: NDArray) -> NDArray:
    """**********ARGUMENTS**********
    :param x: incoming values in numpy array
    **********RETURNS**********
    :return: evaluation (single val for input_sample (row) of x)
    """
    return np.maximum(0, x)


@activation
def relu_leaky(x: NDArray, alpha=0.1) -> NDArray:
    """**********ARGUMENTS**********
    :param x: incoming values in numpy array
    **********RETURNS**********
    :return: evaluation (single val for input_sample (row) of x)
    """
    return np.where(x > 0, x, alpha * x)


# ===================== and their derivatives ======================

@derivative
def sigmoid_derivative(gradient: NDArray, x: Optional[NDArray]=None) -> NDArray:
    """**********ARGUMENTS**********
    :param x: incoming values in numpy array
    **********RETURNS**********
    :return: evaluation (single val for input_sample (row) of x)
    """
    stabalized = np.where(gradient >= 0,
                   1 / (1 + np.exp(-gradient)),
                   np.exp(gradient) / (1 + np.exp(gradient)))

    return stabalized * (1 - stabalized)


@derivative
def relu_derivative(gradient: NDArray, x: Optional[NDArray]) -> NDArray:
    """**********ARGUMENTS**********
    :param x: incoming values in numpy array
    **********RETURNS**********
    :return: evaluation (single val for input_sample (row) of x)
    """
    grad_prime = copy.copy(gradient)
    return np.where(grad_prime < 0, 0, 1)


@derivative
def relu_leaky_derivative(
    gradient: NDArray, x: Optional[NDArray], alpha: float = 0.1
) -> NDArray:
    """**********ARGUMENTS**********
    :param x: incoming values in numpy array
    **********RETURNS**********
    :return: evaluation (single val for val of x)
    """
    grad_prime = np.ones_like(gradient)
    return np.where(grad_prime < 0, alpha, 1)


@derivative
def tanh_derivative(gradient: NDArray, x: Optional[NDArray]) -> NDArray:
    """**********ARGUMENTS**********
    :param x: incoming values in numpy array
    **********RETURNS**********
    :return: evaluation (single val for input_sample (row) of x)
    """
    return 1 - gradient**2


@derivative
def swish_derivative(gradient: NDArray, x: Optional[NDArray]) -> NDArray:
    """
     f(y) = y + σ(x) * (1-y)
     # f'(x) = σ(x) + x * σ(x)(1 - σ(x))
    Parameters
    ----------
    x :

    Returns
    -------

    """
    # TODO: fix this --

    return gradient + x * sigmoid_derivative(gradient)


@derivative
def linear_derivative(gradient: NDArray, x: Optional[NDArray]) -> NDArray:
    """**********ARGUMENTS**********
    :param x: incoming values in numpy array
    **********RETURNS**********
    :return: value of 1 (1st derivative of linear func x = 1)
    """
    return np.ones_like(gradient)


@derivative
def softmax_derivative(gradient: NDArray, x: Optional[NDArray]) -> NDArray:
    """**********ARGUMENTS**********
    :param x: incoming values in numpy array
    **********RETURNS**********
    :return: evaluation
    """

    # identity = np.eye(gradient.shape[1])
    # jacobian = gradient.T * (identity - gradient)
    #
    # jacobian -= np.outer(gradient, gradient)
    # # np.diagflat(self.value) - np.dot(SM, SM.T)

    jacobian = np.diag(gradient) - np.outer(gradient, gradient)

    return jacobian


if __name__ == "__main__":
    x = np.array([[1, 2, 3, 4, 5], [3, 2, 4, 5, 6], [9, 8, 7, 6, 5]]) / 9
    for name, _func in activation_dictionary.items():
        x_hat = _func(x)
        print(name)
        _x = derivative_dictionary[name](gradient=x_hat, x=x)
