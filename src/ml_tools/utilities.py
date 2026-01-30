"""
Utility functions for ml_tools package --- helpers, etc.
"""

import time
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Callable
from datetime import datetime, timedelta
from functools import lru_cache, wraps


def standardize_data(design_matrix: NDArray, axis=0):
    """
    zero tge mean and variance = 1 along axis
    Parameters
    ----------
    design_matrix :
    axis :

    Returns
    -------

    """
    array_mean = np.mean(design_matrix, axis=axis)
    array_std = np.std(design_matrix, axis=axis)

    return (design_matrix - array_mean) / array_std


# -------------- Decorators  --------------
def timed_lru_cache(seconds: int, maxsize: int = 128):
    """from realpython example - uses as @timed_lru_cache"""

    def wrapper_cache(func):
        func = lru_cache(maxsize=maxsize)(func)
        func.lifetime = timedelta(seconds=seconds)
        func.expiration = datetime.utcnow() + func.lifetime

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            if datetime.utcnow() >= func.expiration:
                func.cache_clear()
                func.expiration = datetime.utcnow() + func.lifetime

            return func(*args, **kwargs)

        return wrapped_func

    return wrapper_cache


def log_timeit(func: Callable, logger: Optional):
    """
    to use with a created log object
    Parameters
    ----------
    func :
    logger :

    Returns
    -------

    """

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        message = (
            f"Call of {func.__name__}{args} ({kwargs}) Took {total_time:.4f} seconds"
        )
        if not logger:
            print(message)
        else:
            logger.info(message)

        return result

    return timeit_wrapper


def preformat_expected_shapes(x: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
    """conform the shapes of input and targets, and recast into lower-precision dtype"""
    x_prime = np.asarray(x).astype(np.float16)
    y_prime = np.asarray(y).astype(np.float16)

    if x.ndim == 1:  # single sample
        x_prime = x_prime.reshape(1, -1)
    if y.ndim == 1:
        y_prime = y_prime.reshape(-1, 1)

    assert x_prime.shape[-1] == y_prime.shape[-1]

    return x_prime, y_prime


def rolling_windows_nd(data: NDArray,
                       window_size: int,
                       num_overlap: int = 0,
                       axis: int = 0,
                       ) -> NDArray:
    """
    Given a data array, create overlapping, "rolling" windows using numpy stride tricks
    If we provide some data that is (num_samples, sequence, embedding), such as text data
    eg (20, 90, 64)
    and window the 1st axis (sequence) with a window size = 10, and overlap = 2
    We end with the same data, but with expanded dimensions:
    (20, 11, 10, 64), or (num_samples, num_windows, window_size, embedding_size)
    Parameters
    ----------
    data : data array, of at least 2 dimensions
    window : the size (int) of the window
    axis : the target dimension to tile / roll our windows over.  The specified dimension will be expanded into NxM
        dimensions, where Nis the number of windows, and M is the window size.
    num_overlap : the number of indices to overlap each window

    Returns
    -------

    """
    data_shape = data.shape
    target_length = data_shape[axis]

    if num_overlap > window_size:
        print('rollingWindows: num_overlap > window, so setting to window-1')
        num_overlap = window_size - 1 # shift by one

    shift_length = window_size - num_overlap
    num_windows = np.ceil((target_length - window_size + 1) / shift_length).astype(int)

    new_shape = np.insert(
        np.delete(data_shape, axis),
        axis,
        values=[num_windows, window_size]
    )
    # new shape - (batch, window_count, window_size, latent_space)

    # strides = data.strides[:-1] + (data.strides[-1] * num_Shift, data.strides[-1])
    _leading = data.strides[:axis]
    target = [data.strides[axis] * shift_length, data.strides[axis]]  # expanded to 2D
    _trail = data.strides[axis+1:]

    strides = (*_leading, *target, *_trail)

    windowed_indices = np.lib.stride_tricks.as_strided(data, shape=new_shape, strides=strides)

    return windowed_indices
