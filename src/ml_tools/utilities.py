"""
Utility functions for __package__
"""

import time
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Callable
from datetime import datetime, timedelta
from functools import lru_cache, wraps


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
