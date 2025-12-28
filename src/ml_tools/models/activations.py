import numpy as np
from numpy.typing import NDArray


def sigmoid_activation(x_array: NDArray) -> NDArray:
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


def softmax_activation(x_array: NDArray) -> NDArray:
    """softmaxin' it"""
    exps = np.exp(x_array - np.max(x_array, axis=-1, keepdims=True))
    return exps / np.sum(exps, axis=-1, keepdims=True)


def linear_activation(x_array: NDArray) -> NDArray:
    """linear forward - for regression"""
    return x_array
