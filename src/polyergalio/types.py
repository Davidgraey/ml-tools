"""
Typing complex objects & definitions for __package__
"""
import copy
import inspect
import logging
import warnings
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Tuple,
    TypedDict,
    Union,
)

import numpy as np
from polyergalio.models.constants import EPSILON
from numpy.typing import ArrayLike, NDArray

log = logging.getLogger(__name__)


# -------------- Transform Base Class  --------------
class BasalTransform(ABC):
    """
    Base for stateful transforms: things that learn a mapping from data and
    then apply it. Deliberately narrower than BasalModel -- a transform has no
    forward pass and no loss to minimise on its own behalf, so it only
    promises fit, predict and the two combined.
    """

    _structural_state_keys: tuple[str, ...] = ()

    @abstractmethod
    def fit(self, x_data: NDArray, **kwargs):
        """learn the transform's parameters from data"""

    @abstractmethod
    def predict(self, x_data: NDArray, **kwargs):
        """apply the fitted transform"""

    @abstractmethod
    def fit_predict(self, x_data: NDArray, **kwargs):
        """fit then apply, in one call"""

    @property
    def is_fitted(self) -> bool:
        return getattr(self, "_is_fitted", False)

    @property
    def info(self) -> dict:
        info = {"self": self.__class__}
        info.update(self.__dict__)
        return info

    # --------------- State snapshot / restore ---------------
    def _capture_state(self) -> dict:
        """deep-copy the fields that fully determine this transform's fitted state"""
        return {key: copy.deepcopy(getattr(self, key)) for key in self._structural_state_keys}

    def _restore_state(self, state: dict) -> None:
        """restore a snapshot taken by _capture_state"""
        for key, value in state.items():
            setattr(self, key, value)


# -------------- Model Base Class  --------------
def count_array_elements(value) -> int:
    """Total size of every array nested in a dict, list or tuple."""
    if isinstance(value, np.ndarray):
        return value.size
    if isinstance(value, dict):
        return sum(count_array_elements(member) for member in value.values())
    if isinstance(value, (list, tuple)):
        return sum(count_array_elements(member) for member in value)
    return 0


class BasalModel(ABC):
    _structural_state_keys: tuple[str, ...] = ("x_means", "x_stds", "num_seen_samples")
    registry_name: Optional[str] = None
    training: bool = True
    _registry: dict[str, type] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = cls.registry_name or cls.__name__
        if name in BasalModel._registry and BasalModel._registry[name] is not cls:
            warnings.warn(f"model name {name} redefined; keeping the latest class")
        BasalModel._registry[name] = cls

    def __init__(self,
                 input_dimension: int = 1,
                 output_dimension: int = 1,
                 seed: int = 42):
        self.seed = seed
        self.RNG = np.random.default_rng(seed)

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

        self.x_means = None
        self.x_stds = None
        self.num_seen_samples: int = 0
        self._is_fitted = False

    @abstractmethod
    def forward(self, x_data: NDArray, **kwargs):
        pass

    @abstractmethod
    def predict(self, x_data: NDArray, **kwargs):
        pass

    @abstractmethod
    def fit(self, x_data: NDArray, **kwargs):
        pass

    @abstractmethod
    def fit_predict(self, x_data: NDArray, **kwargs):
        pass

    @abstractmethod
    def calculate_loss(self, **kwargs) -> NDArray:
        pass

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def info(self):
        return ""

    @property
    def num_parameters(self) -> int:
        learned = {
            key: getattr(self, key)
            for key in self._structural_state_keys
            if key not in BasalModel._structural_state_keys
        }
        return count_array_elements(learned)

    def train(self, mode: bool = True) -> "BasalModel":
        """
        Switch between training and inference.

        Returns
        -------
        self, so calls chain: model.eval().predict(x)
        """
        self.training = mode
        return self

    def eval(self) -> "BasalModel":
        """Switch to inference, see train()."""
        return self.train(False)

    # --------------- Serialization ---------------
    def get_config(self) -> dict:
        """Constructor hyperparameters, read from the class signature by name."""
        parameters = inspect.signature(self.__class__.__init__).parameters
        return {
            name: getattr(self, name)
            for name in parameters
            if name != "self" and hasattr(self, name)
        }

    def get_weights(self, for_serialize: bool = False) -> dict:
        """
        Fields named in _structural_state_keys.

        Parameters
        ----------
        for_serialize : return a deep copy rather than live references
        """
        state = {key: getattr(self, key) for key in self._structural_state_keys}
        return copy.deepcopy(state) if for_serialize else state

    def set_weights(self, weights: dict) -> None:
        """Restore fields from get_weights() output and mark the model fitted."""
        for key, value in weights.items():
            setattr(self, key, value)
        self._is_fitted = True

    def serialize(self) -> dict:
        """
        Package identity, hyperparameters and fitted state into a plain dict.
        Config and weights are deep copied, so the result is a snapshot.
        """
        return {
            "type": self.registry_name or self.__class__.__name__,
            "config": copy.deepcopy(self.get_config()),
            "weights": self.get_weights(for_serialize=True),
        }

    @classmethod
    def deserialize(cls, serialized_dict: dict) -> "BasalModel":
        """
        Rebuild a model from serialize() output. Saved config is filtered
        against the current constructor, and dropped keys raise a warning.
        """
        model_cls = cls._registry.get(serialized_dict["type"])
        if model_cls is None:
            raise KeyError(
                f"no model registered as {serialized_dict['type']!r}. Known: {sorted(cls._registry)}"
            )

        accepted = inspect.signature(model_cls.__init__).parameters
        config = {k: v for k, v in serialized_dict["config"].items() if k in accepted}
        dropped = set(serialized_dict["config"]) - set(config)

        if dropped:
            warnings.warn(
                f"{serialized_dict['type']}: dropping saved config keys: {sorted(dropped)}"
            )

        model = model_cls(**config)
        model.set_weights(serialized_dict["weights"])
        return model

    # --------------- Standardization / Normalize ---------------
    def update_running_standardize(self, new_data_mean, new_data_std, new_data_count) -> None:
        """
        proportionally update the running mean and standard deviation for standardization processes
        Parameters
        ----------
        new_data_mean : mean of the new observations or samples under considerations
        new_data_std : standard deviation of the new observations or samples under considerations
        new_data_count : the number of new samples (for proportionally weighting)

        Returns
        -------
        No returns - we update the self. params with the updated mean of the MEAN and STD DEV
        """
        full_count = self.num_seen_samples + new_data_count
        full_mean = (self.num_seen_samples  * self.x_means + new_data_count * new_data_mean) / full_count
        var1 = self.x_stds ** 2
        var2 = new_data_std ** 2

        # error sum of squares
        sum_square_errors = var1 * (self.num_seen_samples  - 1) + var2 * (new_data_count - 1)
        # total group sum of squares
        sum_squares = (self.x_means - full_mean) ** 2 * self.num_seen_samples  + (new_data_mean - full_mean) ** 2 * new_data_count
        full_var = (sum_square_errors + sum_squares) / (full_count - 1)
        full_std = np.sqrt(full_var)

        self.x_means = full_mean
        self.x_stds = full_std

    def standardize(self, data_array: NDArray) -> NDArray:
        """
        Standardize our data array
        Parameters
        ----------
        data_array : numpy array of x-variable

        Returns
        -------
        the mean and standard deviation of the data array
        """
        if self.x_means is None or data_array.shape[-1] != self.x_means.shape[0]:
            log.error("initialize process hasn't been done yet!")

        return (data_array - self.x_means) / (self.x_stds + EPSILON)

    def unstandardize(self, data_array: NDArray) -> NDArray:
        """
        unstandardize the data -> convert back into unit space
        Parameters
        ----------
        data_array : numpy array of x-variable

        Returns
        -------
        the data, transformed back into the original unit space
        """
        assert data_array.shape[-1] == self.x_means.shape[0]

        return data_array * (self.x_stds + EPSILON) + self.x_means

    def init_standardize(self, x_data: NDArray) -> None:
        """
        initalize the standardize variables for tracking, or update them if
        we're adjusting an already fitted model
        Parameters
        ----------
        x_data : NDArray
        """
        if (self.x_means is None) or (self.num_seen_samples == 0):
            # set up the initial values for the new incoming data
            self.num_seen_samples = x_data.shape[0]
            self.x_means = np.mean(x_data, axis=0)
            self.x_stds = np.std(x_data, axis=0)
        else:
            # update the running standardization parameters with proportional weighting
            self.update_running_standardize(
                new_data_mean=np.mean(x_data, axis=0),
                new_data_std=np.std(x_data, axis=0),
                new_data_count=x_data.shape[0])
        pass
