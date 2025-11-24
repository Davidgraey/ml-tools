import pygame as pg
import numpy as np
import entity.entity as entity


# ------------------------------ Game Class -------------------
class Game:
    """
    Class for the Main Game Loop
    """
    def __init__(self, background_path, width, height, title):
        self.title = title

        self.width = width
        self.height = height

        self.game_screen = pg.display.set_mode((self.width, self.height))
        self.game_screen.fill(white_color)

        pg.display.set_caption(self.title)

        background_image = pg.image.load(background_path)

        self.background_image = pg.transform.scale(background_image, (self.width, self.height))

        self.entities = {}

        ''' using .get(key, default_value)
        d = {3: 9, 4: 2}
        default = 99
        print d.get(3, default)
        '''

        self.tick_rate = 60

        # package data for RL agent~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ## self.current_state = []

    def run_game_loop(self) -> None:
        """
        Class for running the game loop
        Returns
        -------

        """

        self.frame_counter = 0

        game_over = False
        complete_level = False

        v_direction = 0
        h_direction = 0

        # Create Objects~~~~~~~~~~~~~~~~~~~~~~~
        # player_character = PlayerCharacter('assets/Robo', 375, 700, 50, 50)

        # NPC objects
        # goal = GameObject('assets/Coin', 375, 50, 50, 50)

        # enemy_0 = EnemyObject('assets/Enemy', 400, 150, 75, 75, 5)
        # enemy_1 = EnemyObject('assets/Enemy', 500, 350, 50, 50, 2)
        # enemy_2 = EnemyObject('assets/Enemy', 600, 500, 75, 75, 5)

        # enemy_list = [enemy_0]  # , enemy_1, enemy_2]
        # goal_list = [goal]

        # Main Loop~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        while not game_over:
            # TODO: blow this out into its own methods
            # EVENTS - Keyboard input
            for event in pg.event.get():
                if event.type == pg.quit:
                    game_over = True

                # key down - begin change in movement
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_UP:
                        v_direction = 'up'
                    elif event.key == pg.K_DOWN:
                        v_direction = 'down'
                    if event.key == pg.K_LEFT:
                        h_direction = 'left'
                    elif event.key == pg.K_RIGHT:
                        h_direction = 'right'
                    # if q is pressed, quit game
                    elif event.key == pg.K_q:
                        game_over = True

                # key up event - halt change in movement
                elif event.type == pg.KEYUP:
                    key_ups = [pg.K_UP, pg.K_DOWN, pg.K_LEFT, pg.K_RIGHT]
                    if event.key in key_ups:
                        h_direction = 0
                        v_direction = 0
            # EVENTS - Actions - input from RL Agent-------
            # No need to use the event que in pg, let's leave that alone and make
            # another set of action / events to directly assign the direction!

            self.game_screen.blit(self.background_image, (0, 0))

            # ---------- Position Updates ----------
            # create self.update_positions

            update_positions(goal_list, self.width, self.game_screen, self.frame_counter)

            # ---------- Position Updates ----------
            player_character.move(v_direction, h_direction, self.height)
            player_character.draw(self.game_screen, player_character.animations[(v_direction, h_direction)],
                                  self.frame_counter)

            update_positions(enemy_list, self.width, self.game_screen, self.frame_counter)

            # ---------- Collisions ----------
            # TODO: create methods for collision checking.
            if player_character.check_collision(enemy_list):
                complete_level = False
                game_over = True
                text = font.render('loss function error', True, black_color)
                self.game_screen.blit(text, (300, 350))
                pg.display.update()

                pg.time.delay(40)
                clock.tick()
                break

            elif player_character.check_collision(goal_list):
                complete_level = True
                game_over = True
                text = font.render('optimized!', True, black_color)
                self.game_screen.blit(text, (300, 350))
                pg.display.update()
                pg.time.delay(40)
                clock.tick()
                break

            else:
                pg.display.update()
                clock.tick(self.tick_rate)

            self.frame_counter += 1

        if complete_level:
            return (states, reinforcement)
            self.run_game_loop()

        else:
            return (states, reinforcement)
            self.run_game_loop()

        pg.quit()

    def build_state(self) -> np.array:
        """
        Build the current state of entities

        Returns
        -------

        """

        #

    def observe_states(self) -> None:
        """
        For each entity, gets global observations
        Determines the next action

        Returns
        -------

        """
        snapshot = self.build_state()

        for ent_id, ent_obj in self.entities.items():
            print(f'operating on {ent_id}, pg object is {ent_obj}')
            # gather observations - we will need to pass in some way...
            ent_obj.gather_observations()

            # take actions - this will make each Entity take the action based on it's observations.
            ent_obj.take_actions()



if __name__ == '__main__':

    # Game parameters~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    screen_width = 800
    screen_height = 800
    screen_title = 'Game Title'

    white_color = (255, 255, 255)
    black_color = (5, 5, 5)

    clock = pg.time.Clock() # can I move this into the class?

    pg.init()
    font = pg.font.SysFont('times', 50, bold=True)

    new_game = Game('assets/background.png', screen_width, screen_height, screen_title)

    state, r = new_game.run_game_loop()