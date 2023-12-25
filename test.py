import pygame
from pygame.locals import * 


import gymnasium as gym
env = gym.make("CarRacing-v2", continuous=False, render_mode="human")

env.reset()

action = 0
while True:
    env.render()
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

    observation, reward, done, truncated, info = env.step(action)
