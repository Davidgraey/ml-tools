"""
docstring for file
__package__
"""
# type: ignore

import time
import logging
from enum import Enum, auto
from typing import Protocol, Tuple, Dict, List, Optional, Union, Iterable, Callable
from typing_extensions import TypedDict

from dataclasses import dataclass, field
from abc import ABC, ABCMeta

from functools import partial, reduce, lru_cache, wraps

import matplotlib.pyplot as plt
from matplotlib import colormaps

# ======= project specifics =======
import numpy as np
from numpy.typing import ArrayLike, NDArray

# -------------------------------- set up logging --------------------------------
log = logging.getLogger(__name__)
log.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
f_handler = logging.FileHandler(f'{__name__}.log')
f_handler.setLevel(logging.INFO)
log.addHandler(f_handler)

RNG = np.random.Gendrator(seed=1)

if __name__ == "__main__":
    print('hello')
