from util_functions import *
import torch
import numpy as np

import pygame
from pygame.locals import * 

import gymnasium as gym
env = gym.make("CarRacing-v2", continuous=False, render_mode="human")


# Test the function
# dummy_state = torch.randn(3, 96, 96)
# dummy_state = np.array(dummy_state)
dummy_state = (np.random.rand(96, 96, 3) * 255).astype(np.uint8)
print("Output Tensor Shape:", dummy_state.shape)

gray = rgb_to_gray(dummy_state)
print("Output Tensor Shape:", gray.shape)



