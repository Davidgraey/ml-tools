import numpy as np
import pygame as pg
import random
import os
from typing import List, Optional, Tuple
from dataclasses import dataclass
# ------ entity senses - local py ------
import senses


# TODO: cartesian to angular and angular to cartesian should be inverses - unit test this
def cartesian_motion_to_angular(cartesian_delta: tuple) -> float:
    """
    Converts an x,y point into angular motion (degrees) and speed
    Parameters
    ----------
    cartesian_delta : point difference, motion translated into (delta_x, delta_y))

    Returns
    -------
    Tuple of (heading, speed), wherein heading is the degrees
    """
    # unpack x and y values
    x,y = cartesian_delta
    speed = np.sqrt(x ** 2 + y ** 2)
    heading = np.arctan2(x, y)
    return (heading, speed)


def angular_motion_to_cartesian(heading: float, speed: float) -> float:
    """
    Converts the angular motion (heading (angle degrees) and speed (float value)) into horizontal and vertical
    Parameters
    ----------
    heading : float, angular heading or facing in degrees
    speed : float, current velocity/speed

    Returns
    -------
    tuple (x_delta, y_delta) - the total change that an object traveling at SPEED along HEADING angle translated into
    cartesian (x,y) coordiantes
    """
    x_delta = speed * np.sin(heading)
    y_delta = speed * np.cos(heading)
    return (x_delta, y_delta)


def get_facing_angle(center_point: np.array, facing_point: np.array, as_radians = False) -> float:
    """
    From the center point and directional indicator facing point, return the angle
    Parameters
    ----------
    center_point : array, [x_coordinate, y_coordinate]
    facing_point : array, [x_coordinate, y_coordinate]
    as_radians: boolean, True to return the facing angle in radians, rather than degrees

    Returns
    -------
    the angle of facing, in degrees, as a float
    """
    _delta = facing_point - center_point
    radians = np.arctan2(_delta[0], _delta[1])
    if as_radians:
        return radians
    return radians_to_degrees(radians)


def degrees_to_radians(degrees: float) -> float:
    """
    Convert degrees to radians
    Parameters
    ----------
    degrees :

    Returns
    -------
    float - the converted angular measurement, given in radians
    """
    return degrees * (np.pi / 180)


def radians_to_degrees(radians: float) -> float:
    """
    Convert radians into degree measurements
    Parameters
    ----------
    radians : float, the original value

    Returns
    -------
    float - the converted angular measurement, given in degrees
    """
    return radians * (180 / np.pi)


# ---------------------------------------------------------------------------------------------------
# TODO: break GameObject into a more basal version
# TODO: experiment with @dataclasses for different aspects of the gameobject? such as facing or movement -then the
#  dataclass could hold some conceptual functions and verifications without mucking up the GameObject -


class GameObject:
    """ Basal class of a gameobject that can move and do cool shit """
    def __init__(self, mass: int, position: Tuple[int, int], size: Tuple[int, int], facing_point: Tuple[int, int]):
        """

        Parameters
        ----------
        mass : mass of entity
        position : x, y coordinates of object's center(?) point
        size : width, height
        facing_point : x, y coordinates that represent the "facing" point; direction
        """
        self.idstring = random.randint(1, 999)

        # initialize position ----------------
        self.position = np.array(position)
        self.facing_point = np.array(facing_point)

        # Constants ----------------
        self.max_speed = 20  # 20 - defined by entity type
        self.speed = 10  # 2 - defined by entity type - aslo the acceleration
        self.mass = mass
        self.size = size

        # Collisions ----------------
        self.bounding_box = pg.rect.Rect((self.position), self.size)
        self.bounding_box.center = self.position # this 'center' point, or the x and y, may need to be adjusted

        # Constantly Updated Variables
        self.old_speed = (0, 0)

    # ------------------------  Visualize ------------------------

    def render(self):
        surface = pg.Surface(self.size)
        surface = surface.fill(color=(0,0,0,0))
        surface = self._draw_sense(surface)
        surface = self._draw_self(surface)

        # canvas.blit(self, self.position)
        # canvas.blit(sprites, (self.x_position, self.y_position))  # draw single sprite
        return surface, self.position

    def _draw_sense(self, surface):
        # composite and return the sensory objects
        # TODO: senses - build and draw

        # example ----
        pg.draw.circle(surface=surface,
                       color=(255, 255, 0, 50),
                       center=self.position,
                       radius=max(self.size[0], self.size[1]) * 1.5,
                       width=0)

        return surface

    def _draw_self(self, surface):
        # composite and return entity object
        # example ----
        pg.draw.circle(surface=surface,
                       color=(255, 255, 0, 125),
                       center=self.position,
                       radius=max(self.size[0], self.size[1]) // 2,
                       width=4, # width of > 0 - stroke
                       draw_top_left=True)

        # surface.blit()
        return surface

    # ------------------------ MOVEMENT ------------------------

    def move(self, screen_width, x_move: int, y_move: int) -> None:
        # move self.position
        self.position = self.position + [x_move, y_move]
        self.facing_point = self.facing_point + [x_move, y_move]

    @property
    def direction(self):
        # angular direction in degrees
        return get_facing_angle(self.position, self.facing_point)

    # @property
    # def position(self):
    #     return (self.x_position, self.y_position)

    def __repr__(self):
        return f'Object at {self.position} facing {self.direction} \n has mass of {self.mass} and is size {self.size}'


#--------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------


class Entity(GameObject):
    """
    What all does an entity need?
        - standard game object params
        - sensory system (with confidences or jitter) feeds into:
        - Decision system
        - Action system
    """
    def __init__(self, id):
        """

        Parameters
        ----------
        id :
        """
        super().__init__(mass=, position=, size=, image_path=)
        self.id = id
        # mass
        # position
        # velocity
        # max_force
        # max_speed
        # orientation
            # vector - center point -> projected point for 2d, 2 for 3D?

        # size
            # object size in space
            # bounding

        # img
