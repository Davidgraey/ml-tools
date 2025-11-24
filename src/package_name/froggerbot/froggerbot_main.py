"""
Froggerbot - main runnable loop
"""
import pygame as pg
import numpy as np
import copy
import entity.entity as entity


class Menu:
    ###
class Message:
    ###


# ------------------------------ Game Class --------------------------------------
class Game:
    def __init__(self, height=600, width=800):
        """

        Parameters
        ----------
        height :
        width :
        """
        # Initalize pygame ------------------------
        pg.init()

        # Clock and frames ------------------------
        self.clock = pg.time.Clock()
        self.frame_counter = 0
        self.tick_rate = 60

        # States ------------------------
        self.running = True
        self.h_direction = 0
        self.v_direction = 0

        # Visualizing elements ------------------------
        self.game_screen = pg.display.set_mode((width, height))
        self.game_screen.fill(pg.Color("floralwhite"))

        # Container ------------------------
        self.pc_entity = entity.Entity()
        self.entity_list = []
        self.game_state = []  # TODO: find a way to pack the game state;

    def process_input(self) -> None:
        """  """
        for event in pg.event.get():
            if event.type == pg.quit:
                self.running = False
                break

            # key down - begin change in movement
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_UP:
                    self.v_direction += 1
                elif event.key == pg.K_DOWN:
                    self.v_direction += -1
                if event.key == pg.K_LEFT:
                    self.h_direction += -1
                elif event.key == pg.K_RIGHT:
                    self.h_direction += 1
                # if q is pressed, quit game
                elif event.key == pg.K_q:
                    self.running = False
                    break

            # key up event - halt change in movement
            elif event.type == pg.KEYUP:
                key_ups = [pg.K_UP, pg.K_DOWN, pg.K_LEFT, pg.K_RIGHT]
                if event.key in key_ups:
                    self.h_direction = 0
                    self.v_direction = 0

    def pack_state(self):
        """ text here """
        # [e.position for e in self.entity_list]
        pass

    def update(self):
        """ text here """
        self.pc_entity.update(position=(self.h_direction, self.v_direction))
        for _ent in self.entity_list:
            _ent.update()
        pass

    def render(self) -> None:
        """ text here """
        # TODO: in future: cache un-updated items?
        game_screen = copy.copy(self.game_screen)
        game_screen = self._draw_background(surface = game_screen)
        game_screen = self._draw_entities(surface = game_screen)
        game_screen = self._draw_pc_entity(surface = game_screen)
        game_screen = self.draw_effects(surface = game_screen)

        self.game_screen = game_screen

    def _draw_background(self, surface):
        """ s """
        # draw each layer of background
        pass

    def _draw_entities(self, surface):
        """ s """
        # TODO: make a list comp and build up a layer
        for _ent in self.entity_list:
            _ent.draw()
        pass

    def _draw_pc_entity(self, surface):
        """ s """

        pass

    def _draw_effects(self, surface):
        """ s """
        pass

    def run_primary_loop(self) -> None:
        """ text here """

        while self.running:
            self.process_input()
            self.update()
            self.render()
            self.clock.tick(60)


# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------


# class GameState():
#     def __init__(self):
#         self.x = 120
#         self.y = 120
#
#     def update(self,moveCommandX,moveCommandY):
#         self.x += moveCommandX
#         self.y += moveCommandY
#
# class UserInterface():
#     def __init__(self):
#         pg.init()
#         self.window = pg.display.set_mode((640,480))
#         pg.display.set_caption("Discover Python & Patterns - https://www.patternsgameprog.com")
#         self.clock = pg.time.Clock()
#         self.gameState = GameState()
#         self.running = True
#         self.moveCommandX = 0
#         self.moveCommandY = 0
#
#     def processInput(self):
#         self.moveCommandX = 0
#         self.moveCommandY = 0
#         for event in pg.event.get():
#             if event.type == pg.QUIT:
#                 self.running = False
#                 break
#             elif event.type == pg.KEYDOWN:
#                 if event.key == pg.K_ESCAPE:
#                     self.running = False
#                     break
#                 elif event.key == pg.K_RIGHT:
#                     self.moveCommandX = 8
#                 elif event.key == pg.K_LEFT:
#                     self.moveCommandX = -8
#                 elif event.key == pg.K_DOWN:
#                     self.moveCommandY = 8
#                 elif event.key == pg.K_UP:
#                     self.moveCommandY = -8
#
#     def update(self):
#         self.gameState.update(self.moveCommandX,self.moveCommandY)
#
#     def render(self):
#         self.window.fill((0,0,0))
#         x = self.gameState.x
#         y = self.gameState.y
#         pg.draw.rect(self.window,(0,0,255),(x,y,400,240))
#         pg.display.update()
#
#     def run(self):
#         while self.running:
#             self.processInput()
#             self.update()
#             self.render()
#             self.clock.tick(60)

if __name__ == "__main__":

game_obj = Game()
game_obj.run_game_loop()

pg.quit()