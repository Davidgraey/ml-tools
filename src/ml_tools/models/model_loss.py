
import numpy as np
import copy
from numpy.typing import NDArray
from typing import Optional
from ml_tools.models.constants import ClassificationTask, determine_classification_task


def cross_entropy(logits: NDArray,
                  targets: NDArray,
                  task: ClassificationTask) -> float|NDArray:
    """
    CROSS ENTROPY - stabalized versions to accept logit values
    Assumes that targets are one-hot encoded vector [0, 0, 1, 0]
    Parameters
    ----------
    logits : numpy.ndarray  the forward-pass logit values
    targets : numpy.ndarray   Assumes that targets are one-hot encoded vector [0, 0, 1, 0]Assumes that targets are one-hot encoded vector [0, 0, 1, 0]
    task : ClassificationTask enum

    Returns
    -------
    numpy.float64
    """
    #  multilabel, multinomial classification with logits ------
    if task == ClassificationTask.MULTILABEL:
        log_sum_exp = np.log(1 + np.exp(logits))
        loss = log_sum_exp - targets * logits

    #  categorical cross-entropy with stabilization -- using log-softmax of the logits -------
    elif task == ClassificationTask.MULTINOMIAL:
        # shift our values (0-ceiling) for numeric stability
        shifted_logits = logits - np.max(logits, axis=-1, keepdims=True)
        # log-sum-exp for numerical stability
        log_sum_exp = np.log(np.sum(np.exp(shifted_logits), axis=-1))
        # targets becomes a mask to select the true class logit
        # class_logits = shifted_logits[targets.astype(bool)]
        # loss = -class_logits + log_sum_exp
        cls = np.argmax(targets, axis=-1)
        correct = shifted_logits[np.arange(logits.shape[0]), cls]
        loss = -correct + log_sum_exp

    # Binary cross-entropy -- logit stabilizing for neg and positive --------
    elif task == ClassificationTask.BINARY:
        loss = np.maximum(0, logits) - (targets * logits) + np.log(1 + np.exp(-np.abs(logits)))

    return np.mean(loss)


def cross_entropy_derivative(prediction: NDArray,
                             targets: NDArray,
                             task: Optional[ClassificationTask]) -> NDArray | float:
    """ BACKPROP TRICKS for sigmoid / softmax: combine"""
    sample_count = targets.shape[0]
    return (prediction - targets)  / sample_count
