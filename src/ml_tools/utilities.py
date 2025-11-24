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


def update_mean_std(mean1, std1, count1, mean2, std2, count2):
    '''**********ARGUMENTS**********
    :param mean1: self.data_type_means
    :param std1: self.data_type_stds
    :param count1: num_before
    :param mean2: this_data_mean
    :param std2: this_data_std
    :param count2: num_new
    **********RETURNS**********
    :return: new mean value, new std value
    '''
    # from www.burtonsys.com/climate/composite_sd.php#python
    countBoth = count1 + count2
    meanBoth = (count1 * mean1 + count2 * mean2) / countBoth
    var1 = std1 ** 2
    var2 = std2 ** 2
    # error sum of squares
    ESS = var1 * (count1 - 1) + var2 * (count2 - 1)
    # total group sum of squares
    TGSS = (mean1 - meanBoth) ** 2 * count1 + (mean2 - meanBoth) ** 2 * count2
    varBoth = (ESS + TGSS) / (countBoth - 1)
    stdBoth = np.sqrt(varBoth)

    return meanBoth, stdBoth


def standardize(X, means, stds):
    '''**********ARGUMENTS**********
    :param X: data to be standardized
    :param means: mean of sample and previous samples gotten from update_mean_std
    :param stds: std of sample and previous samples gotten from update_mean_std
    '**********RETURNS**********
    :return: standardized data
    '''
    return (X - means) / (stds + EPSILON)


def unstandardize(X, means, stds):
    '''**********ARGUMENTS**********
    :param X: data to be returned to normal space
    :param means: mean of sample and previous samples gotten from update_mean_std
    :param stds: std of sample and previous samples gotten from update_mean_std
    '**********RETURNS**********
    :return: unstandardized data
    '''
    return (stds - EPSILON) * X + means