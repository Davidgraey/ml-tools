'''
ERROR and LOSS FUNCTIONS all done in Numpy--
'''
import numpy as np
from numpy.typing import NDArray
from enum import Enum
from typing import Optional
from ml_tools.models.constants import ClassificationTask, Reductions, determine_classification_task
from ml_tools.models.clustering.plsom_utils import cosine_distance
import copy


loss_dictionary, derivative_dictionary = {}, {}

loss_func = lambda f: loss_dictionary.setdefault(f.__name__, f)
derivative = lambda f: derivative_dictionary.setdefault(f.__name__, f)


@loss_func
def difference(prediction: NDArray,
               targets: NDArray,
               axis: int = 0,
               reduction: Optional[Reductions]=None,
               task: Optional[ClassificationTask]=None) -> NDArray:
    """
    Calculate the difference between prediction and targets; with reduction

    Parameters
    ----------
    prediction : input array, should be of shape [n_respondents, number_items]
    targets :
    axis :
    reduction :

    Returns
    -------
    array of transformed values as numpy array, with the same shape as prediction input
    """
    diff = np.abs(targets - prediction)
    if reduction == 'mean':
        return np.mean(diff, axis=axis)
    elif reduction == 'sum':
        return np.sum(diff, axis=axis)
    return diff


@loss_func
def mse(prediction: NDArray,
        targets: NDArray,
        axis: int = 0,
        reduction: Optional[Reductions] = None,
        task: Optional[ClassificationTask] = None) -> (float|NDArray):
    """
    MEAN SQUARED ERRORS FOR REGRESSION MODELS
    Parameters
    ----------
    prediction : numpy array of predictions (outputs)
    targets : True or Target values - dimensionally match prediction

    Returns
    -------
    evaluation of error / loss
    """
    # constant adjusted loss to make derivative clearner
    return 0.5 * np.mean((prediction - targets) ** 2)


@loss_func
def rmse(prediction: NDArray,
         targets: NDArray,
         axis: int = 0,
         reduction: Optional[Reductions] = None,
         task: Optional[ClassificationTask] = None) -> NDArray:
    """
    Calculate the root-mean-square of the data -

    Parameters
    ----------
    prediction : input array, should be of shape [n_respondents, number_items]
    targets :
    axis :
    reduction :

    Returns
    -------
    array of transformed values as numpy array, with the same shape as prediction input
    """
    rmse = np.sqrt(np.mean((targets - prediction) ** 2, axis=axis))
    if reduction == 'mean':
        return np.mean(rmse)
    elif reduction == 'sum':
        return np.sum(rmse)
    return rmse

@loss_func
def sse(prediction: NDArray, targets: NDArray, axis: int = 0, reduction: Optional[Reductions]=None, task: Optional[
    ClassificationTask]=None) -> NDArray:
    """
    Calculate the sum of squared errors

    Parameters
    ----------
    prediction : input array, should be of shape [n_respondents, number_items]
    targets :
    axis :
    reduction :

    Returns
    -------
    array of transformed values as numpy array, with the same shape as prediction input
    """
    rmse = np.sum((targets - prediction) ** 2, axis=axis)
    if reduction == 'mean':
        return np.mean(rmse)
    elif reduction == 'sum':
        return np.sum(rmse)
    return rmse


@loss_func
def mae(prediction: NDArray,
        targets: NDArray, axis: int = 0,
        reduction: Optional[Reductions]=None,
        task: Optional[ClassificationTask]=None) -> NDArray:
    """
    Calculate the mean-absolute-error of the data -

    Parameters
    ----------
    prediction : input array, should be of shape [n_respondents, number_items]
    targets :
    axis :
    reduction :

    Returns
    -------
    array of transformed values as numpy array, with the same shape as prediction input
    """
    mae = np.mean(np.abs((targets - prediction), axis=axis))
    if reduction == 'mean':
        return np.mean(mae)
    elif reduction == 'sum':
        return np.sum(mae)
    return mae

@loss_func
def cosine_error(prediction: NDArray, targets: NDArray, axis: int = 0, reduction: Optional[Reductions]=None, task: Optional[
    ClassificationTask]=None) -> NDArray:
    """
    squared error for cosine distance loss

    Parameters
    ----------
    prediction : input array, should be of shape [n_respondents, number_items]
    targets :
    axis :
    reduction :

    Returns
    -------
    array of transformed values as numpy array, with the same shape as prediction input
    """
    cos = cosine_distance(prediction, targets)
    if reduction == 'mean':
        return np.mean(cos)
    elif reduction == 'sum':
        return np.sum(cos)
    return cos


@loss_func
def cross_entropy(prediction: NDArray,
                  targets: NDArray,
                  reduction: Optional[Reductions]=None,
                  task: Optional[ClassificationTask]=None) -> float|NDArray:
    """
    CROSS ENTROPY - stabalized versions to accept logit values
    Assumes that targets are one-hot encoded vector [0, 0, 1, 0]
    Parameters
    ----------
    prediction : numpy.ndarray  the forward-pass logit values
    targets : numpy.ndarray   Assumes that targets are one-hot encoded vector [0, 0, 1, 0]Assumes that targets are one-hot
    encoded vector [0, 0, 1, 0]
    task : ClassificationTask enum

    Returns
    -------
    numpy.float64
    """
    #  multilabel, multinomial classification with prediction ------
    if task == ClassificationTask.MULTILABEL:
        log_sum_exp = np.log(1 + np.exp(prediction))
        loss = log_sum_exp - targets * prediction

    #  categorical cross-entropy with stabilization -- using log-softmax of the prediction -------
    elif task == ClassificationTask.MULTINOMIAL:
        # shift our values (0-ceiling) for numeric stability
        shifted_logits = prediction - np.max(prediction, axis=-1, keepdims=True)
        # log-sum-exp for numerical stability
        log_sum_exp = np.log(np.sum(np.exp(shifted_logits), axis=-1))
        # targets becomes a mask to select the true class logit
        # class_logits = shifted_logits[targets.astype(bool)]
        # loss = -class_logits + log_sum_exp
        cls = np.argmax(targets, axis=-1)
        correct = shifted_logits[np.arange(prediction.shape[0]), cls]
        loss = -correct + log_sum_exp

    # Binary cross-entropy -- logit stabilizing for neg and positive --------
    elif task == ClassificationTask.BINARY:
        loss = np.maximum(0, prediction) - (targets * prediction) + np.log(1 + np.exp(-np.abs(prediction)))

    return np.mean(loss)

#------------------------------------------------------------------
@derivative
def mse_derivative(prediction, targets, **kwargs) -> float|NDArray:
    return (prediction - targets)


@derivative
def mae_derivative(prediction: NDArray, targets: NDArray) -> NDArray:
    return (prediction - targets) / np.abs(prediction - targets)


@derivative
def rmse_derivative(prediction: NDArray, targets: NDArray) -> NDArray:
    return np.abs(targets - prediction) / np.sqrt(prediction.shape[0])


@derivative
def cross_entropy_derivative(prediction: NDArray,
                             targets: NDArray,
                            **kwargs) -> NDArray | float:
    """ BACKPROP TRICKS for sigmoid / softmax: combine"""
    sample_count = targets.shape[0]
    return (prediction - targets)  / sample_count
