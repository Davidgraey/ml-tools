"""
ERROR and LOSS FUNCTIONS all done in Numpy--
"""

import numpy as np
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from typing import Optional
from ml_tools.models.activations import sigmoid, softmax
from ml_tools.models.constants import ClassificationTask, Reductions
from ml_tools.distances import cosine_distance
from ml_tools.models.constants import EPSILON


loss_dictionary, derivative_dictionary = {}, {}

loss_func = lambda f: loss_dictionary.setdefault(f.__name__, f)
derivative = lambda f: derivative_dictionary.setdefault(f.__name__, f)


class Loss(ABC):
    def __init__(self):
        self.targets = None
        self.prediction = None
        self.mask_column = None
        self.valid_elements = None

    @staticmethod
    def _mask_column(mask: Optional[NDArray], reference: NDArray) -> Optional[NDArray]:
        """
        Normalize a mask to broadcast against an elementwise array.

        Parameters
        ----------
        mask : (..., 1) or (...,) with 1 for real content, 0 for padding
        reference : the elementwise array the mask will multiply

        Returns
        -------
        (..., 1) mask, or None if mask is None
        """
        if mask is None:
            return None
        positions = mask[..., 0] if mask.ndim == reference.ndim else mask
        return positions[..., None]

    @staticmethod
    def _position_mask(mask: Optional[NDArray], reference: NDArray) -> Optional[NDArray]:
        """
        Normalize a mask to match an already-reduced, per-position array.

        Parameters
        ----------
        mask : (..., 1) or (...,) with 1 for real content, 0 for padding
        reference : an array one axis narrower than the raw prediction,
            e.g. a per-position loss after reducing over the class axis

        Returns
        -------
        mask matching reference's shape, or None if mask is None
        """
        if mask is None:
            return None
        positions = mask[..., 0] if mask.ndim == reference.ndim + 1 else mask
        return positions.astype(reference.dtype)

    @staticmethod
    def _valid_elements(mask_column: NDArray, reference: NDArray) -> float:
        return max(mask_column.sum() * reference.shape[-1], 1.0)

    @abstractmethod
    def forward(self, prediction: NDArray, targets: NDArray, mask: Optional[NDArray] = None) -> float | NDArray:
        pass

    @abstractmethod
    def backward(self) -> NDArray:
        """
        Returns dL / d(y_pred)
        """
        pass

    def __call__(self, predictions: NDArray, targets: NDArray, mask: Optional[NDArray] = None) -> float | NDArray:
        return self.forward(predictions, targets, mask)


class DifferenceLoss(Loss):
    def forward(self, prediction, targets, mask: Optional[NDArray] = None):
        self.prediction = prediction
        self.targets = targets
        diff = np.abs(targets - prediction)
        self.mask_column = self._mask_column(mask, diff)
        return diff * self.mask_column if self.mask_column is not None else diff

    def backward(self):
        grad = np.sign(self.prediction - self.targets)
        return grad * self.mask_column if self.mask_column is not None else grad


class MSELoss(Loss):
    def forward(self, prediction, targets, mask: Optional[NDArray] = None):
        self.prediction = prediction
        self.targets = targets
        diff = prediction - targets
        self.mask_column = self._mask_column(mask, diff)

        if self.mask_column is not None:
            self.valid_elements = self._valid_elements(self.mask_column, diff)
            return np.sum((diff ** 2) * self.mask_column) / self.valid_elements
        return np.mean(diff ** 2)

    def backward(self):
        """forward means over every element, so the reduction is size not shape[0]"""
        diff = self.prediction - self.targets
        if self.mask_column is not None:
            return (2 / self.valid_elements) * diff * self.mask_column
        return (2 / self.prediction.size) * diff


class RMSELoss(Loss):
    def forward(self, prediction, targets, mask: Optional[NDArray] = None):
        self.prediction = prediction
        self.targets = targets
        diff = prediction - targets
        self.mask_column = self._mask_column(mask, diff)

        if self.mask_column is not None:
            self.valid_elements = self._valid_elements(self.mask_column, diff)
            self.rmse = np.sqrt(np.sum((diff ** 2) * self.mask_column) / self.valid_elements)
        else:
            self.rmse = np.sqrt(np.mean(diff ** 2))
        return self.rmse

    def backward(self):
        diff = self.prediction - self.targets
        if self.mask_column is not None:
            return diff * self.mask_column / (self.valid_elements * (self.rmse + EPSILON))
        N = diff.size
        return diff / (N * (self.rmse + EPSILON))


class SSELoss(Loss):
    def forward(self, prediction, targets, mask: Optional[NDArray] = None):
        self.prediction = prediction
        self.targets = targets
        diff = prediction - targets
        self.mask_column = self._mask_column(mask, diff)
        if self.mask_column is not None:
            return np.sum((diff ** 2) * self.mask_column)
        return np.sum(diff ** 2)

    def backward(self):
        grad = 2 * (self.prediction - self.targets)
        return grad * self.mask_column if self.mask_column is not None else grad


class MAELoss(Loss):
    def forward(self, prediction, targets, mask: Optional[NDArray] = None):
        self.prediction = prediction
        self.targets = targets
        diff = prediction - targets
        self.mask_column = self._mask_column(mask, diff)

        if self.mask_column is not None:
            self.valid_elements = self._valid_elements(self.mask_column, diff)
            return np.sum(np.abs(diff) * self.mask_column) / self.valid_elements
        return np.mean(np.abs(diff))

    def backward(self):
        diff = self.prediction - self.targets
        if self.mask_column is not None:
            return np.sign(diff) * self.mask_column / self.valid_elements
        return np.sign(diff) / diff.size


class CosineLoss(Loss):
    def forward(self, prediction, targets, mask: Optional[NDArray] = None):
        self.prediction = prediction
        self.targets = targets

        dot = np.sum(prediction * targets, axis=-1)
        norm_p = np.linalg.norm(prediction, axis=-1)
        norm_t = np.linalg.norm(targets, axis=-1)

        self.cos = dot / (norm_p * norm_t + 1e-8)
        per_position = 1 - self.cos
        self.position_mask = self._position_mask(mask, per_position)

        if self.position_mask is not None:
            self.valid_elements = max(self.position_mask.sum(), 1.0)
            return np.sum(per_position * self.position_mask) / self.valid_elements

        self.valid_elements = per_position.size
        return np.sum(per_position) / self.valid_elements

    def backward(self):
        p = self.prediction
        t = self.targets

        norm_p = np.linalg.norm(p, axis=-1, keepdims=True)
        norm_t = np.linalg.norm(t, axis=-1, keepdims=True)

        grad = (
            p * np.sum(p * t, axis=-1, keepdims=True) / (norm_p**3 * norm_t)
            - t / (norm_p * norm_t)
        )
        if self.position_mask is not None:
            grad = grad * self.position_mask[..., None]
        return grad / self.valid_elements


class CrossEntropyLoss(Loss):
    def __init__(self, task):
        super().__init__()
        self.task = task

    def forward(self, prediction, targets, mask: Optional[NDArray] = None):
        self.prediction = prediction
        self.targets = targets

        if self.task == ClassificationTask.MULTINOMIAL:
            shifted = prediction - np.max(prediction, axis=-1, keepdims=True)
            log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1))
            cls = np.argmax(targets, axis=-1)
            correct = np.take_along_axis(shifted, cls[..., None], axis=-1)[..., 0]
            loss = -correct + log_sum_exp
            self.reduce_mask = self._position_mask(mask, loss)
            self.valid_elements = (
                max(self.reduce_mask.sum(), 1.0) if self.reduce_mask is not None else loss.size
            )

        elif self.task == ClassificationTask.BINARY:
            loss = (
                np.maximum(0, prediction)
                - targets * prediction
                + np.log(1 + np.exp(-np.abs(prediction)))
            )
            self.reduce_mask = self._mask_column(mask, loss)
            self.valid_elements = (
                self._valid_elements(self.reduce_mask, loss) if self.reduce_mask is not None else loss.size
            )

        elif self.task == ClassificationTask.MULTILABEL:
            log_sum_exp = np.log(1 + np.exp(prediction))
            loss = log_sum_exp - targets * prediction
            self.reduce_mask = self._mask_column(mask, loss)
            self.valid_elements = (
                self._valid_elements(self.reduce_mask, loss) if self.reduce_mask is not None else loss.size
            )

        if self.reduce_mask is not None:
            return np.sum(loss * self.reduce_mask) / self.valid_elements
        return np.mean(loss)

    def backward(self):
        """
        dL / d(logits). forward() consumes logits and applies its own
        log-softmax / log-sigmoid, so the squashing belongs here too.
        """
        if self.task == ClassificationTask.MULTINOMIAL:
            grad = softmax(self.prediction) - self.targets
            if self.reduce_mask is not None:
                grad = grad * self.reduce_mask[..., None]
            return grad / self.valid_elements

        grad = sigmoid(self.prediction) - self.targets
        if self.reduce_mask is not None:
            grad = grad * self.reduce_mask
        return grad / self.valid_elements

    def __call__(self, predictions: NDArray, targets: NDArray, mask: Optional[NDArray] = None):
        return self.forward(predictions, targets, mask)


class MultiHeadLoss(Loss):
    def __init__(self, losses: list[Loss], weights=None):
        super().__init__()
        self.losses = losses
        self.weights = weights or [1.0] * len(losses)

    def forward(self, predictions: NDArray, targets, mask: Optional[list] = None):
        self.predictions = predictions
        self.targets = targets
        masks = mask if mask is not None else [None] * len(self.losses)

        total = 0.0
        self.last_losses = []
        for w, loss, yhat, y, m in zip(self.weights, self.losses, predictions, targets, masks):
            L = loss.forward(yhat, y, m)
            self.last_losses.append(L)
            total += w * L
        return total

    def backward(self):
        return [w * loss.backward() for w, loss in zip(self.weights, self.losses)]

    def __call__(self, predictions: NDArray, targets: NDArray, mask: Optional[list] = None):
        return self.forward(predictions, targets, mask)


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
