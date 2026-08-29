from abc import ABC, abstractmethod
import numpy as np
from ml_tools.models.layers import Layer

class Optimizer(ABC):
    """
    Abstract base class for all optimizers.
    """

    def __init__(self):
        pass

    @abstractmethod
    def step(self, layers: list[Layer]) -> None:
        """
        Take one step of the optimizer function

        Parameters
        ----------
        layers : layers (List[Layer]): the ORDERED LIST of model structure

        """
        pass


    def zero_gradients(self, layers: list[Layer]):
        """
        We'll have to set all of our gradients to zero
        Useful if gradients are accumulated.
        """
        for layer in layers:
            layer.zero_gradients()


class SGD(Optimizer):
    """
    STOCHASTIC GRADIENT DESCENT - as vanilla as we can get
    """
    def __init__(self, learning_rate: float = 0.001, clip_gradients: bool = False):
        """

        Parameters
        ----------
        learning_rate : our learning rate, or alpha
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.max_norm = 1.0
        self.do_clipping = clip_gradients  # TODO: fix this


    def _scale(self, value):
        """
        Gradient dictionaries nest as deep as the layers do -- a block holding a
        block holding a layer -- so the scaling recurses rather than assuming
        one level.
        """
        if isinstance(value, dict):
            return {key: self._scale(sub) for key, sub in value.items()}
        return self.learning_rate * value

    def step(self, layers: list[Layer]) -> None:

        for layer in layers:
            delta_grads = layer.get_gradients()
            if not delta_grads:
                continue

            # matching keys via unpacking delta gradients
            layer.update_weights(
                **{key: self._scale(sub) for key, sub in delta_grads.items()}
            )


# ADAM


# Scaled Conjugate Gradient
