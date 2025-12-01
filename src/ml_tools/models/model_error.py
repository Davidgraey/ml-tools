'''
ERROR FUNCTIONS all done in Numpy--
'''
import numpy as np
from numpy.typing import NDArray
from enum import Enum
from ml_tools.models.plsom_utils import cosine_distance

class Reductions(Enum):
    MEAN = 'mean'
    SUM = 'sum'
    NONE = None

def difference(x: NDArray, y: NDArray, axis: int = 0, reduction: Reductions=None) -> NDArray:
    """
    Calculate the difference between x and y; with reduction

    Parameters
    ----------
    x : input array, should be of shape [n_respondents, number_items]
    y :
    axis :
    reduction :

    Returns
    -------
    array of transformed values as numpy array, with the same shape as x input
    """
    diff = np.abs(x - y)
    if reduction == 'mean':
        return np.mean(diff)
    elif reduction == 'sum':
        return np.sum(diff)
    return diff


def rmse(x: NDArray, y: NDArray, axis: int = 0, reduction: Reductions=None) -> NDArray:
    """
    Calculate the root-mean-square of the data -

    Parameters
    ----------
    x : input array, should be of shape [n_respondents, number_items]
    y :
    axis :
    reduction :

    Returns
    -------
    array of transformed values as numpy array, with the same shape as x input
    """
    rmse = np.sqrt(np.mean((x-y) ** 2, axis=axis))
    if reduction == 'mean':
        return np.mean(rmse)
    elif reduction == 'sum':
        return np.sum(rmse)
    return rmse


def mae(x: NDArray, y: NDArray, axis: int = 0, reduction: Reductions=None) -> NDArray:
    """
    Calculate the mean-absolute-error of the data -

    Parameters
    ----------
    x : input array, should be of shape [n_respondents, number_items]
    y :
    axis :
    reduction :

    Returns
    -------
    array of transformed values as numpy array, with the same shape as x input
    """
    mae = np.mean(np.abs((x-y), axis=axis))
    if reduction == 'mean':
        return np.mean(mae)
    elif reduction == 'sum':
        return np.sum(mae)
    return mae


def cosine_error(x: NDArray, y: NDArray, axis: int = 0, reduction: Reductions=None) -> NDArray:
    """
    squared error for cosine distance loss

    Parameters
    ----------
    x : input array, should be of shape [n_respondents, number_items]
    y :
    axis :
    reduction :

    Returns
    -------
    array of transformed values as numpy array, with the same shape as x input
    """
    cos = cosine_distance(x, y)
    if reduction == 'mean':
        return np.mean(cos)
    elif reduction == 'sum':
        return np.sum(cos)
    return cos
