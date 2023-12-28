import pygame
from pygame.locals import * 

import gymnasium as gym
env = gym.make("CarRacing-v2", continuous=False, render_mode="human")

state, _ = env.reset()


action = 0
while True:
    env.render()
    print(state.shape)
    keys = pygame.key.get_pressed()
    if keys[K_RIGHT]:
        action = 1
    elif keys[K_LEFT]:
        action = 2
    elif keys[K_UP]:
        action = 3
    elif keys[K_DOWN]:
        action = 4
    elif keys[K_SPACE]:
        env.reset()
    else:
        action = 0


    # observation: Box(0, 255, (96, 96, 3), uint8)
    # reward: float -> the reward you get at each time step, Total reward calculation is 1000 - 0.1*N where N is timestep count
    # terminated: bool - If car visits all tiles or if reward is -100
    # truncated: bool - if car is going backwards
    # info: dict - empty
    state, reward, terminated, truncated, info = env.step(action)
    
    


