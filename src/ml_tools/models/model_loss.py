"""
ERROR and LOSS FUNCTIONS all done in Numpy--
"""

import numpy as np
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from typing import Optional
from ml_tools.models.constants import ClassificationTask, Reductions
from ml_tools.models.distances import cosine_distance


loss_dictionary, derivative_dictionary = {}, {}

loss_func = lambda f: loss_dictionary.setdefault(f.__name__, f)
derivative = lambda f: derivative_dictionary.setdefault(f.__name__, f)


class Loss(ABC):
    def __init__(self):
        self.targets = None
        self.prediction = None

    @abstractmethod
    def forward(self, prediction: NDArray, targets: NDArray) -> float | NDArray:
        pass

    @abstractmethod
    def backward(self) -> NDArray:
        """
        Returns dL / d(y_pred)
        """
        pass

    def __call__(self, predictions: NDArray, targets: NDArray) -> float | NDArray:
         return self.forward(predictions, targets)


class DifferenceLoss(Loss):
    def forward(self, prediction, targets):
        self.prediction = prediction
        self.targets = targets
        return np.abs(targets - prediction)

    def backward(self):
        return np.sign(self.prediction - self.targets)


class MSELoss(Loss):
    def forward(self, prediction, targets):
        self.prediction = prediction
        self.targets = targets
        diff = prediction - targets
        return 0.5 * np.mean(diff**2)

    def backward(self):
        return self.prediction - self.targets


class RMSELoss(Loss):
    def forward(self, prediction, targets):
        self.prediction = prediction
        self.targets = targets
        diff = prediction - targets
        self.rmse = np.sqrt(np.mean(diff ** 2))
        return self.rmse

    def backward(self):
        diff = self.prediction - self.targets
        N = diff.size
        return diff / (N * (self.rmse + EPSILON))


    class SSELoss(Loss):
        def forward(self, prediction, targets):
            self.prediction = prediction
            self.targets = targets
            return np.sum((prediction - targets) ** 2)

        def backward(self):
            return 2 * (self.prediction - self.targets)


class MAELoss(Loss):
    def forward(self, prediction, targets):
        self.prediction = prediction
        self.targets = targets
        return np.mean(np.abs(prediction - targets))

    def backward(self):
        diff = self.prediction - self.targets
        return np.sign(diff) / diff.size


class CosineLoss(Loss):
    def forward(self, prediction, targets):
        self.prediction = prediction
        self.targets = targets

        dot = np.sum(prediction * targets, axis=-1)
        norm_p = np.linalg.norm(prediction, axis=-1)
        norm_t = np.linalg.norm(targets, axis=-1)

        self.cos = dot / (norm_p * norm_t + 1e-8)
        return np.mean(1 - self.cos)

    def backward(self):
        p = self.prediction
        t = self.targets

        norm_p = np.linalg.norm(p, axis=-1, keepdims=True)
        norm_t = np.linalg.norm(t, axis=-1, keepdims=True)

        grad = (
            p * np.sum(p * t, axis=-1, keepdims=True) / (norm_p**3 * norm_t)
            - t / (norm_p * norm_t)
        )
        return grad / p.shape[0]


class CrossEntropyLoss(Loss):
    def __init__(self, task):
        super().__init__()
        self.task = task

    def forward(self, prediction, targets):
        self.prediction = prediction
        self.targets = targets

        if self.task == ClassificationTask.MULTINOMIAL:
            shifted = prediction - np.max(prediction, axis=-1, keepdims=True)
            log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1))
            cls = np.argmax(targets, axis=-1)
            correct = shifted[np.arange(prediction.shape[0]), cls]
            loss = -correct + log_sum_exp

        elif self.task == ClassificationTask.BINARY:
            loss = (
                np.maximum(0, prediction)
                - targets * prediction
                + np.log(1 + np.exp(-np.abs(prediction)))
            )

        elif self.task == ClassificationTask.MULTILABEL:
            log_sum_exp = np.log(1 + np.exp(prediction))
            loss = log_sum_exp - targets * prediction

        return np.mean(loss)

    def backward(self):
        sample_count = self.targets.shape[0]
        return (self.prediction - self.targets) / sample_count

    def __call__(self, predictions: NDArray, targets: NDArray):
         return self.forward(predictions, targets)


class MultiHeadLoss(Loss):
    """
    h  = fc1.forward(x)

    y1_hat = fc2.forward(h)
    y2_hat = fc3.forward(h)

    L = loss.forward(
        preds=[y1_hat, y2_hat],
        targets=[y1, y2]
    )
    # BACKPROPIGATION =========
    # loss --> heads
    g_y1, g_y2 = loss.backward()

    # heads --> shared
    g_h1 = fc2.backward(g_y1)
    g_h2 = fc3.backward(g_y2)

    # merge aka sum gradients from both heads
    g_h = g_h1 + g_h2

    # input
    fc1.backward(g_h)
    """
    def __init__(self, losses: list[Loss], weights=None):
        super().__init__()
        self.losses = losses
        # if we're reweighting the losses:
        self.weights = weights or [1.0] * len(losses)

    def forward(self, predictions: NDArray, targets):
        self.predictions = predictions
        self.targets = targets

        total = 0.0
        self.last_losses = []

        for w, loss, yhat, y in zip(self.weights, self.losses, predictions, targets):
            L = loss.forward(yhat, y)
            self.last_losses.append(L)
            total += w * L

        return total

    def backward(self):
        grads = []

        for w, loss in zip(self.weights, self.losses):
            grads.append(w * loss.backward())

        return grads

    def __call__(self, predictions: NDArray, targets: NDArray):
         return self.forward(predictions, targets)

# TODO: retire the functions in favor of OOP version
@loss_func
def difference(
    prediction: NDArray,
    targets: NDArray,
    axis: int = 0,
    reduction: Optional[Reductions] = None,
    task: Optional[ClassificationTask] = None,
) -> NDArray:
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
    if reduction == "mean":
        return np.mean(diff, axis=axis)
    elif reduction == "sum":
        return np.sum(diff, axis=axis)
    return diff


@loss_func
def mse(
    prediction: NDArray,
    targets: NDArray,
    axis: int = 0,
    reduction: Optional[Reductions] = None,
    task: Optional[ClassificationTask] = None,
) -> float | NDArray:
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
def rmse(
    prediction: NDArray,
    targets: NDArray,
    axis: int = 0,
    reduction: Optional[Reductions] = None,
    task: Optional[ClassificationTask] = None,
) -> NDArray:
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
    if reduction == "mean":
        return np.mean(rmse)
    elif reduction == "sum":
        return np.sum(rmse)
    return rmse


@loss_func
def sse(
    prediction: NDArray,
    targets: NDArray,
    axis: int = 0,
    reduction: Optional[Reductions] = None,
    task: Optional[ClassificationTask] = None,
) -> NDArray:
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
    if reduction == "mean":
        return np.mean(rmse)
    elif reduction == "sum":
        return np.sum(rmse)
    return rmse


@loss_func
def mae(
    prediction: NDArray,
    targets: NDArray,
    axis: int = 0,
    reduction: Optional[Reductions] = None,
    task: Optional[ClassificationTask] = None,
) -> NDArray:
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
    if reduction == "mean":
        return np.mean(mae)
    elif reduction == "sum":
        return np.sum(mae)
    return mae


@loss_func
def cosine_error(
    prediction: NDArray,
    targets: NDArray,
    axis: int = 0,
    reduction: Optional[Reductions] = None,
    task: Optional[ClassificationTask] = None,
) -> NDArray:
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
    if reduction == "mean":
        return np.mean(cos)
    elif reduction == "sum":
        return np.sum(cos)
    return cos


@loss_func
def cross_entropy(
    prediction: NDArray,
    targets: NDArray,
    reduction: Optional[Reductions] = None,
    task: Optional[ClassificationTask] = None,
) -> float | NDArray:
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
        loss = (
            np.maximum(0, prediction)
            - (targets * prediction)
            + np.log(1 + np.exp(-np.abs(prediction)))
        )

    return np.mean(loss)


# ------------------------------------------------------------------
@derivative
def mse_derivative(prediction, targets, **kwargs) -> float | NDArray:
    return prediction - targets


@derivative
def mae_derivative(prediction: NDArray, targets: NDArray) -> NDArray:
    return (prediction - targets) / np.abs(prediction - targets)


@derivative
def rmse_derivative(prediction: NDArray, targets: NDArray) -> NDArray:
    return np.abs(targets - prediction) / np.sqrt(prediction.shape[0])


@derivative
def cross_entropy_derivative(
    prediction: NDArray, targets: NDArray, **kwargs
) -> NDArray | float:
    """BACKPROP TRICKS for sigmoid / softmax: combine"""
    sample_count = targets.shape[0]
    return (prediction - targets) / sample_count
