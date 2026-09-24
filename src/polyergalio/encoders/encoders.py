"""
------------------------ ABC for our encoder / processors ------------------------------
"""
from abc import ABC, abstractmethod


class Processor(ABC):
    def __init__(self, target: (str | int), variable_idx: int):
        self.target = target
        self.variable_idx = variable_idx
        self._fitted: bool = False
        self.obs_min_max: tuple = None

    @abstractmethod
    def inverse(self, values):
        """inverse of the encoding function"""

    @abstractmethod
    def encode(self, values):
        """encoding of the values"""

    @abstractmethod
    def fit_encode(self, values):
        """fit and encoding of the  values"""

    @abstractmethod
    def fit(self, values):
        """fit the encoder obj using the data"""

    @property
    @abstractmethod
    def metadata(self):
        """gets the metadata of the encoder to be passed to backbone network"""

    @property
    def is_fitted(self):
        """simple bool - check if the encoder has been fitted to data"""
        return self._fitted

    @property
    def info(self):
        info = {"self": self.__class__}
        info.update(self.__dict__)
        return info

    @property
    def additional_targets(self):
        return None

    def __repr__(self):
        return (
            f"encoder of processor {self.__class__} targeting {self.target} at {self.variable_idx}"
        )

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __getstate__(self):
        # Return a dictionary that defines what to pickle
        state = self.__dict__.copy()  # Copy the current state
        # defining this here in case we need specific behaviors to be included
        return state
