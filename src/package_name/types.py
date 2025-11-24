"""
Typing complex objects & definitions for __package__
"""
from enum import Enum, auto
from typing import Protocol, Tuple, Dict, List, Optional, Union, Iterable, Callable
from typing_extensions import TypedDict


# -------------- Complex typing  --------------
class ComplexType(TypedDict):
    attribute_1: int
    # ...
