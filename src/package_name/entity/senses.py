import numpy as np
import pygame as pg
from dataclasses import dataclass
from enum import Enum

# dataclass constructor for different types?
# enums for type & shape

class SensorType(Enum):
    """ SENSORY MODALITIES """
    SIGHT = 1
    HEARING = 2
    SMELL = 3
    ELECTROMAG = 4
    PRESSURE = 5
    OMNISICENCE = 6


class SensorShape(Enum):
    """ THE SHAPE OF PERCEPTION """
    CIRCLE = 1
    TRIANGLE = 2
    CARDIOD = 3
    BILOBED = 4
    OMNI = 5


@dataclass
class SensorySystem:
    type: str
    shape: int
    range: int
    noise: float
    xray: float

    def visualize(self):
        """
        Returns a visual representation of the sensory system.  Called once during entity init?
        returned object does not need to be redrawn, can be held in the entity object
        draws a "vision cone" for the sense
        Noise - represented by wavy line?
        xray - opacity?
        range - object length
        shape -

        """
        #build up a transparent object to return.

        return False

    def __repr__(self):
        return f'{self.type}'

    def __str__(self):
        return f'{self.type}'

'''
what to hold?
sense type
sense shape - relative to the entity's facing point (ie vision - in front 45 degrees
sense range
sense noise / jitter factor - For example, vision could have less noise but smaller shape / range.  Could also take 
the form of uncertainty? (bayesian nnet?)
sense x-ray (hearing could avoid obstruction while vision would not.) ( could be a factor? electrosense could be full x-ray; hearing 0.6, vision 0.0

All of these could just be factors / values, so dataclass makes sense.
'''

