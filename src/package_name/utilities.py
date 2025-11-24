"""
Utility functions for __package__
"""
import time
from typing import Optional, Callable
from datetime import datetime, timedelta
from functools import lru_cache, wraps


# -------------- Decorators  --------------
def timed_lru_cache(seconds: int, maxsize: int = 128):
    """ from realpython example - uses as @timed_lru_cache"""
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
        message = f'Call of {func.__name__}{args} ({kwargs}) Took' \
                  f' {total_time:.4f} seconds'
        if not logger:
            print(message)
        else:
            logger.info(message)

        return result
    return timeit_wrapper


# TODO: add common operations like reverse_dict, etc.
#