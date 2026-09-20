"""
Typing complex objects & definitions for __package__
"""
import copy
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
from ml_tools.models.constants import EPSILON
from numpy.typing import ArrayLike, NDArray


# -------------- Transform Base Class  --------------
class BasalTransform(ABC):
    """
    Base for stateful transforms: things that learn a mapping from data and
    then apply it. Deliberately narrower than BasalModel -- a transform has no
    forward pass and no loss to minimise on its own behalf, so it only
    promises fit, predict and the two combined.
    """

    # fields every BasalTransform needs snapshotted to fully serialize/restore
    # a fitted instance. Empty here -- unlike BasalModel, BasalTransform has no
    # __init__ and owns no state of its own -- so a subclass defines its own
    # fitted-parameter fields from scratch, the same way it extends this empty
    # tuple rather than overriding it outright.
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
class BasalModel(ABC):
    # fields every BasalModel needs snapshotted to fully serialize/restore a
    # fitted instance -- just the standardization stats here, since BasalModel
    # itself owns no learned parameters of its own. A subclass extends this
    # tuple with whatever holds its own learned state (weights, centroids,
    # ...) rather than overriding it outright, so the standardization stats
    # are never accidentally dropped from a subclass's snapshot.
    _structural_state_keys: tuple[str, ...] = ("x_means", "x_stds")

    def __init__(self,
                 input_dimension: int = 1,
                 output_dimension: int = 1,
                 seed: int = 42):
        self.RNG = np.random.default_rng(seed)

        self.weights_shape: tuple[int, ...] = (input_dimension, output_dimension)
        self.weights = np.empty(self.weights_shape)

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

        self.x_means = None
        self.x_stds = None

        pass

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
    def fit_predict(self,
                    x_data: NDArray,
                    verbose: bool,
                    num_iterations: int,
                    labels: Optional[ArrayLike],
                    **kwargs):
        pass

    @abstractmethod
    def calculate_loss(self, **kwargs) -> NDArray:
        pass

    @property
    def params(self):
        return self.weights

    @property
    def info(self):
        return ""

    # --------------- State snapshot / restore ---------------
    def _capture_state(self) -> dict:
        """deep-copy the fields that fully determine this model's fitted state"""
        return {key: copy.deepcopy(getattr(self, key)) for key in self._structural_state_keys}

    def _restore_state(self, state: dict) -> None:
        """restore a snapshot taken by _capture_state"""
        for key, value in state.items():
            setattr(self, key, value)

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
        try:
            assert data_array.shape[-1] == self.x_means.shape[0]
        except AssertionError:
            log.error(f"initialize process hasn't been done yet!")

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

        return data_array * (self.x_stds - EPSILON) + self.x_means

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



# -------------- Protocols for functional processes  --------------

# Loss protocol
# @Protocol
def basal_loss(prediction: NDArray, targets: NDArray, **kwargs) -> NDArray:
    return

# Activation Protocol
# derivative Protocol


# -------------- Complex typing  --------------
class ComplexType(TypedDict):
    attribute_1: int
    # ...


