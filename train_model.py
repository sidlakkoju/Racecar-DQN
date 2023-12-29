import pygame
from pygame.locals import * 
import gymnasium as gym

from agent import agent
from util_functions import *
from parameters import *

env = gym.make("CarRacing-v2", continuous=False)
agent = agent(frame_stack_num, action_space, learning_rate, memory_size, training_batch_size, discount_factor, epsilon, epsilon_decay, epsilon_min)

agent.load_model("models/model_weights_mps.pth")


for episode in range(episode_count):

    state, _ = env.reset()
    state = rgb_to_gray(state)
    frame_queue = deque(maxlen=frame_stack_num)
    for _ in range(frame_stack_num):
        frame_queue.append(state)
    
    total_reward = 0
    negative_reward_counter = 0
    time_step = 0


    while True:    

        # Get Action
        frame_stack = get_frame_stack(frame_queue) 
        action = agent.get_action_explore(frame_stack)

        # Take Action For Frame Stack
        reward = 0
        for _ in range(frame_stack_num):
            state, r, terminated, truncated, info = env.step(action)
            reward += r
            state = rgb_to_gray(state)
            frame_queue.append(state)
            if render:
                env.render()
        
        # Negative Reward Counter
        if time_step > 100 and reward < 0:
            negative_reward_counter += 1
        
        # Add To Memory
        next_frame_stack = get_frame_stack(frame_queue)
        agent.add_to_memory(frame_stack, action, reward, next_frame_stack, terminated)
        
        # Episode Terminaton Conditions
        if terminated or truncated or negative_reward_counter > 25:
            print("Episode Terminated")
            break
        
        # Replay Learning
        loss = 0
        if len(agent.memory) > training_batch_size:
            loss = agent.replay()        

        agent.save_model("models/model_weights_extra.pth")

        time_step += 1
        total_reward += reward
    print("Episode: {} | Time Step: {} | Action: {} | Total_Reward: {} | Loss: {} | Epsilon {}".format(episode, time_step, action, total_reward, loss, agent.epsilon))
        