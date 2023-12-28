import pygame
from pygame.locals import * 
import gymnasium as gym

from agent import agent
from util_functions import *
from parameters import *

env = gym.make("CarRacing-v2", continuous=False, render_mode="human")
agent = agent(frame_stack_num, action_space, learning_rate, memory_size, training_batch_size, discount_factor, epsilon, epsilon_decay, epsilon_min)
agent.load_model("models/model_weights_mps.pth")


frame_queue = deque(maxlen=frame_stack_num)

state, _ = env.reset()
state = rgb_to_gray(state)
for _ in range(frame_stack_num):
    frame_queue.append(state)
    

while True:
    env.render()    
    frame_stack = get_frame_stack(frame_queue) 
    action = agent.get_action(frame_stack)
    state, reward, terminated, truncated, info = env.step(action)
    state = rgb_to_gray(state)
    frame_queue.append(state)
