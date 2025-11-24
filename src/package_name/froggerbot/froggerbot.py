#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 2019

This is an example of a RL agent in the pygame space - it's an akward
implementation, but may have some things we can cannibalize.
"""

import pygame
import random
import dill as pickle
import numpy as np
import os


def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

def update_positions(object_list, screen_width, canvas, frame_counter):
    '''for each game object in object list, run object.move and object.draw
        this is for NPC objects - enemies and goals'''
    for game_object in object_list:
        game_object.move(screen_width)
        game_object.draw(canvas, game_object.animate_idle, frame_counter)


#------------------------------Game Class----------------#
class Game:
    
    def __init__(self, background_path, width, height, title):
        self.title = title
        self.width = width
        self.height = height
        
        
        self.game_screen = pygame.display.set_mode((self.width, self.height))
        self.game_screen.fill(white_color)
        pygame.display.set_caption(self.title)
        background_image = pygame.image.load(background_path)
        self.background_image = pygame.transform.scale(background_image, (self.width, self.height))
        
        self.tick_rate = 60
        
        #package data for RL agent~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.current_state = []


    def rl_pack_states(self, obj_character, obj_goals, obj_enemy, game_over, complete_level, verbose = False):
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #package data for RL agent~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #we need - game objects - player, enemies, goals - any other object types?
        global states, reinforcement
        
        self.current_state = [[obj_character], obj_goals, obj_enemy]
        self.current_state = [item.position for sublist in self.current_state for item in sublist]
        r = game_over * complete_level # could move up to collision check loops if we want more control
        
        if states == []:
            states = np.array(self.current_state)
            reinforcement = [r]
        else:
            states = np.dstack((states, np.array(self.current_state)))
            reinforcement.append(r)
        
        if verbose:
            print(states.shape, len(reinforcement))
            print(states[..., -1], reinforcement[-1])
        return


    def run_game_loop(self):
        
        self.frame_counter = 0
        
        game_over = False
        complete_level = False
        
        v_direction = 0
        h_direction = 0
        
        #Create Objects~~~~~~~~~~~~~~~~~~~~~~~
        player_character = PlayerCharacter('assets/Robo', 375, 700, 50, 50)
        
        #NPC objects
        goal = GameObject('assets/Coin', 375, 50, 50, 50)
         
        enemy_0 = EnemyObject('assets/Enemy', 400, 150, 75, 75, 5)
        enemy_1 = EnemyObject('assets/Enemy', 500, 350, 50, 50, 2)
        enemy_2 = EnemyObject('assets/Enemy', 600, 500, 75, 75, 5)
        
        enemy_list = [enemy_0]#, enemy_1, enemy_2]
        goal_list = [goal]
        
        #Main Loop~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        while not game_over:
    
            #EVENTS - Keyboard input
            for event in pygame.event.get():
                if event.type == pygame.quit:
                    game_over = True
                
                #key down - begin change in movement
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        v_direction = 'up'
                    elif event.key == pygame.K_DOWN:
                        v_direction = 'down'
                    if event.key == pygame.K_LEFT:
                        h_direction = 'left'
                    elif event.key == pygame.K_RIGHT:
                        h_direction = 'right'
                    #if q is pressed, quit game
                    elif event.key == pygame.K_q:
                        game_over = True
                        
                #key up event - halt change in movement
                elif event.type == pygame.KEYUP:
                    key_ups = [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]
                    if event.key in key_ups:
                        h_direction = 0
                        v_direction = 0
            #EVENTS - Actions - input from RL Agent-------
                #No need to use the event que in pygame, let's leave that alone and make
                #another set of action / events to directly assign the direction!
                #if rl_action = 'up
                
                
                
                #####
            self.game_screen.blit(self.background_image, (0,0))
            
            update_positions(goal_list, self.width, self.game_screen, self.frame_counter)
            
            #movement update
            player_character.move(v_direction, h_direction, self.height)
            player_character.draw(self.game_screen, player_character.animations[(v_direction, h_direction)], self.frame_counter)
            
            update_positions(enemy_list, self.width, self.game_screen, self.frame_counter)
            
            
            if player_character.check_collision(enemy_list):
                complete_level = False
                game_over = True
                text = font.render('loss function error', True, black_color)
                self.game_screen.blit(text, (300, 350))
                pygame.display.update()
                #pack final state - gameplay loop breaks before the previous state package occurs
                self.rl_pack_states(player_character, goal_list, enemy_list, game_over, complete_level, verbose = True)
                pygame.time.delay(40)
                clock.tick()
                break
            
            elif player_character.check_collision(goal_list):
                complete_level = True
                game_over = True
                text = font.render('optimized!', True, black_color)
                self.game_screen.blit(text, (300, 350))
                pygame.display.update()
                #pack final state - gameplay loop breaks before the previous state package occurs
                self.rl_pack_states(player_character, goal_list, enemy_list, game_over, complete_level, verbose = True)
                pygame.time.delay(40)
                clock.tick()
                break
            
            
            #pygame.draw.rect(game_screen, black_color, [350, 350, 100, 100])
            #pygame.draw.circle(game_screen, black_color, [400, 300], 50)
            else:
                pygame.display.update()
                clock.tick(self.tick_rate)
            
            self.frame_counter += 1
            self.rl_pack_states(player_character, goal_list, enemy_list, game_over, complete_level, verbose = True)
            
        print(states.shape, reinforcement.shape)
        
        if complete_level:
            return (states, reinforcement)
            self.run_game_loop()
            
        else:
            return (states, reinforcement)
            self.run_game_loop()
        
        
        stack = [states, reinforcement]
        with open("real_experience_buffer.pickle", "wb") as filename:
            pickle.dump(stack, filename)
        
        pygame.quit()
        #quit()
        


#------------------------------------------------------------------------------------


class GameObject():
    
    def __init__(self, image_path, x, y, width, height):
        #Constant Variables
        self.image_path = image_path
        self.idstring = random.randint(10, 99) #'other' game objects have id of 2 digits
        self.base_sprite = pygame.transform.scale(pygame.image.load(image_path +'/base.png'), (width, height))
        self.bounding_box = self.base_sprite.get_rect().inflate(-4, -2)
        self.bounding_box.center = (x, y)
        
        #Updating Variables
        self.x_position = x
        self.y_position = y
        self.width = width
        self.height = height
        
        #Animation states initialize
        self.animate_idle = []
        self.animate_left = []
        self.animate_up = []
        self.animate_right = []
        self.animate_down = []
        
        self.animate_sprite_init()
        self.animations = {('up', 0) : self.animate_up,
                           (0, 'up') : self.animate_up,
                           ('down', 0) : self.animate_down,
                           (0, 'down') : self.animate_down,
                           ('up', 'right') : self.animate_right,
                           (0, 'right') : self.animate_right,
                           ('up', 'left') : self.animate_left,
                           (0, 'left') : self.animate_left,
                           ('down', 'right') : self.animate_right,
                           ('down', 'left') : self.animate_left,
                           (0, 0) : self.animate_idle
                           }
        
        
    def draw(self, canvas, sprites, frame_counter):
        animation_index = (frame_counter // 5) % 2 # every 5 frames we animate frame [0] or [1]
        if type(sprites) == list:
            canvas.blit(sprites[animation_index], (self.x_position, self.y_position))
        else:
            canvas.blit(sprites, (self.x_position, self.y_position)) #draw single sprite
        
    def move(self, screen_width):
        pass
    
    def animate_sprite_init(self):
        global project_path
        sprites_idle = []
        sprites_left = []
        sprites_up = []
        sprites_right = []
        sprites_down = []
        
        sprite_lists = [sprites_idle, sprites_left, sprites_up, sprites_right, sprites_down]
        loop_type = ['idle','left', 'up', 'right', 'down']
        loop_zip = zip(sprite_lists, loop_type)
        os.chdir(project_path)
         
        for sl, lt in loop_zip:
            if os.path.exists(self.image_path + '/' + lt):
                os.chdir(self.image_path + '/' + lt)
                #make list of filenames in directory
                filenames = [i for i in os.listdir() if i.endswith('png')]
                for image in filenames:
                    #print(image)
                    this_img = pygame.transform.scale(pygame.image.load(image), (self.width, self.height))
                    sl.append(this_img)
                os.chdir(project_path)
            else:
                #print('no folder', lt)
                pass
        
        flipper = lambda x: pygame.transform.flip(x, True, False)
        
        self.animate_idle = sprites_idle
        self.animate_left = list(map(flipper, sprites_right));
        self.animate_up = sprites_up #same image sequence for this game
        
        #pygame.transform.flip(x, True, False)
        self.animate_right = sprites_right
        self.animate_down = sprites_idle
    
    @property
    def position(self):
        return (self.x_position, self.y_position)



class PlayerCharacter(GameObject):
    def __init__(self, image_path, x, y, width, height):
        
        super().__init__(image_path, x, y, width, height)
        
        #Constants
        self.max_speed = 20 #10
        self.speed = 10 #2
        self.idstring = random.randint(1, 9) #pc has id of 1 digit
        self.bounding_box = self.bounding_box.inflate(-6, 0)
        
        #Updating Variables
        self.v_old_speed = 0
        self.h_old_speed = 0
        
        
    def move(self, v_direction, h_direction, max_height):
        v_move_speed = int(self.speed + self.v_old_speed)
        h_move_speed = int(self.speed + self.h_old_speed)
        
        if v_direction == 'up':
            self.y_position -= v_move_speed
            self.v_old_speed = clamp(((.2 * self.speed) + self.v_old_speed), 0, self.max_speed)
            
        elif v_direction == 'down':
            self.y_position += v_move_speed
            self.v_old_speed = clamp(((.2 * self.speed) + self.v_old_speed), 0, (self.max_speed * .666))
        
        else:
            self.v_old_speed = self.v_old_speed * .5
        
        if h_direction == 'left':
            self.x_position -= h_move_speed
            self.h_old_speed = clamp(((.2 * self.speed) + self.h_old_speed), 0, self.max_speed)
            
        elif h_direction == 'right':
            self.x_position += h_move_speed
            self.h_old_speed = clamp(((.2 * self.speed) + self.h_old_speed), 0, self.max_speed)
        
        
        else:
            self.h_old_speed = self.h_old_speed * .5
            
        #print(v_move_speed, h_move_speed)
        #print(self.v_old_speed, self.h_old_speed)
        #Check X and Y Bounds------------------
        #Future - maybe update boundry collision with pygame rec collide
        if self.y_position > max_height - 20: #max y is the bottom of screen
            self.y_position = max_height - 20
        
        elif self.y_position < 20: #min y is the top of screen
            self.y_position = 20
        
        if self.x_position < 300:
            self.x_position = 300
            
        elif self.x_position > 450:
            self.x_position = 450
            
        #update bounding box-----------------
        self.bounding_box.center = (self.x_position, self.y_position)
        
        
    def rl_move(self, direction, max_height):
        v_move_speed = int(self.speed + self.v_old_speed)
        h_move_speed = int(self.speed + self.h_old_speed)
        if direction == 'up':
            self.y_position -= v_move_speed
            self.v_old_speed = clamp(((.2 * self.speed) + self.v_old_speed), 0, self.max_speed)
            
        elif direction == 'down':
            self.y_position += v_move_speed
            self.v_old_speed = clamp(((.2 * self.speed) + self.v_old_speed), 0, (self.max_speed * .666))
        
        else:
            self.v_old_speed *= 0.8
        
        if direction == 'left':
            self.x_position -= h_move_speed
            self.h_old_speed = clamp(((.2 * self.speed) + self.h_old_speed), 0, self.max_speed)
            
        elif direction == 'right':
            self.x_position += h_move_speed
            self.h_old_speed = clamp(((.2 * self.speed) + self.h_old_speed), 0, self.max_speed)
        
        else:
            self.h_old_speed *= 0.8
            
            
        #Check X and Y Bounds------------------
        #Future - maybe update boundry collision with pygame rec collide
        if self.y_position > max_height - 20: #max y is the bottom of screen
            self.y_position = max_height - 20
        
        elif self.y_position < 20: #min y is the top of screen
            self.y_position = 20
        
        if self.x_position < 300:
            self.x_position = 300
            
        elif self.x_position > 450:
            self.x_position = 450
            
        #update bounding box-----------------
        self.bounding_box.center = (self.x_position, self.y_position)
    
    
    def check_collision(self, object_list):
        '''collision check'''
        results = []
        for other_object in object_list:
            results.append(self.bounding_box.colliderect(other_object.bounding_box))
            
        if any(results):
            collider = [i for i,j in zip(object_list, results) if j]
            print('collision with', collider[0].idstring)
        return any(results)
        
        

class EnemyObject(GameObject):
    
    def __init__(self, image_path, x, y, width, height, speed):
        
        super().__init__(image_path, x, y, width, height)
        
        self.speed = speed #random.randint(1, 15)
        self.idstring = random.randint(100, 999) #enemies have an id 3 digits
         
        
    def move(self, width):
        if self.x_position <= 55:
            self.speed = abs(self.speed)
        elif self.x_position >= width - 105:
            self.speed =  -1 * abs(self.speed)
            
        self.x_position += self.speed
        self.bounding_box.center = (self.x_position, self.y_position)

project_path = os.path.abspath('')

if __name__ == '__main__':
    ############################################################################
    #Game parameters~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    screen_width = 800
    screen_height = 800
    screen_title = 'Road-Bot'
    
    
    white_color = (255, 255, 255)
    black_color = (5, 5, 5)
    
    #global container for states
    
    project_path = os.path.abspath('')
    #os.path.abspath('')
    
    ##################
    
    clock = pygame.time.Clock()
    
    pygame.init()
    font = pygame.font.SysFont('times', 50, bold = True)
    states = []
    reinforcement = 0
    
    
    new_game = Game('assets/background.png',screen_width, screen_height, screen_title)
    #per game instance, return state and reinforcement.  Need another loop to encapsulate this game loop, to run x instances of the game and get state/r pairs out.
    
    state, r = new_game.run_game_loop() 
    
    #stack state with states, r into reinforcement.
