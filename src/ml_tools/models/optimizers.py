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
        block holding a layer
        scaling should be applied recursively rather than assuming one level.
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


class Adam(Optimizer):
    """
    Momentum estimates are kept per (layer, parameter name), since a layer can
    expose several independently-shaped parameters -- FullyConnectedLayer's
    weights and bias, for instance, need separate moments.
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        momentum: float = 0.9,
        ridge_momentum: float = 0.999,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_decay = momentum
        self.ridge_decay = ridge_momentum
        self.eps = eps
        self.timestep = 0
        self._momenta: dict = {}
        self._ridges: dict = {}

    def update(self, value, path: tuple, momentum_update: float, ridge_update: float):
        # careful -- recursive
        if isinstance(value, dict):
            return {
                key: self.update(sub, path + (key,), momentum_update, ridge_update)
                for key, sub in value.items()
            }
        if value is None:
            return None

        if path not in self._momenta:
            self._momenta[path] = np.zeros_like(value)
            self._ridges[path] = np.zeros_like(value)
        m = self._momenta[path]
        v = self._ridges[path]

        m = self.momentum_decay * m + (1 - self.momentum_decay) * value
        v = self.ridge_decay * v + (1 - self.ridge_decay) * value ** 2
        self._momenta[path] = m
        self._ridges[path] = v

        m_hat = m / momentum_update
        v_hat = v / ridge_update
        return self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

    def step(self, layers: list[Layer]) -> None:
        self.timestep += 1
        momentum_update = 1 - self.momentum_decay ** self.timestep
        ridge_update = 1 - self.ridge_decay ** self.timestep

        for layer in layers:
            delta_grads = layer.get_gradients()
            if not delta_grads:
                continue

            layer.update_weights(**{key: self.update(sub, (layer, key), momentum_update, ridge_update)
                    for key, sub in delta_grads.items()
                }
            )

# Scaled Conjugate Gradient
