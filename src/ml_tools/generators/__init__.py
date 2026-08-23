"""
description
https://packaging.python.org/en/latest/

"""
from .data_generators import RandomDatasetGenerator
import logging

__version__ = "0.1.0"

# ________ set up logging ________
log = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
f_handler = logging.FileHandler(f'{__name__}.log')
f_handler.setLevel(logging.INFO)
log.addHandler(f_handler)


# ________ set up public objects ________
__all__ = ["__version__"]
