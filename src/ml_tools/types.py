"""
Typing complex objects & definitions for __package__
"""
from enum import Enum, auto
import numpy as np
from numpy.typing import NDArray, ArrayLike
from typing import Protocol, Tuple, Dict, List, Optional, Union, Iterable, Callable, TypedDict
from abc import ABC, abstractmethod
from ml_tools.models.constants import EPSILON


# -------------- Model Base Class  --------------
class BasalModel(ABC):
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
        assert data_array.shape[-1] == self.x_means.shape[0]

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


