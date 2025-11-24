import pygame as pg
import numpy as np
from typing import Tuple

class Background:
    def __init__(self, color, alpha, size: Tuple[int, int]):
        # Build
        self.surface = pg.Surface(size)
        # RGBA
        self.surface.fill(color) # alpha

        pass

    # ------------------------  Visualize ------------------------
    def render(self, canvas):
        # self._draw_self(canvas)

        # canvas.blit(self, self.position)
        # canvas.blit(sprites, (self.x_position, self.y_position))  # draw single sprite
        return canvas

    def _draw_self(self, canvas):
        # composite and return entity object
        # self.surface
        pass


class Tile:
    def __init__(self, color, alpha, size: Tuple[int, int], position: Tuple[int, int]):
        #
        self.surface = pg.Surface(size)
        self.surface.fill(color) # alpha
        self.position = position

    # ------------------------  Visualize ------------------------
    def render(self, canvas):
        # self._draw_self(canvas)

        # canvas.blit(self, self.position)
        # canvas.blit(sprites, (self.x_position, self.y_position))  # draw single sprite
        return canvas

    def _draw_self(self, canvas):
        # composite and return entity object
        # self.surface
        pass

