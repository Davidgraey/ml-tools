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
    def __init__(self, learning_rate: float = 0.001):
        """

        Parameters
        ----------
        learning_rate : our learning rate, or alpha
        """
        super().__init__()
        self.learning_rate = learning_rate

    def step(self, layers: list[Layer]) -> None:
        for layer in layers:
            delta_grads = layer.get_gradients()
            delta_grads = {
                k: self.learning_rate * v
                for k, v in delta_grads.items()
            }
            # matching keys via unpacking delta gradients
            layer.update_weights(**delta_grads)


# ADAM


# Scaled Conjugate Gradient
