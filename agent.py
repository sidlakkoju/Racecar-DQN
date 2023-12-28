import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np 
import random 
class agent:

    def __init__(self, frame_stack_num, action_space, learning_rate, memory_size, training_batch_size, discount_factor, epsilon, epsilon_decay, epsilon_min):
        if torch.cuda.is_available():
            print("CUDA")
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            print("MPS")
            self.device = torch.device("mps")
        else:
            print("CPU")
            self.device = torch.device("cpu")

        self.action_space = action_space    
        self.frame_stack_num = frame_stack_nums
        self.training_batch_size = training_batch_size
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.memory = deque(maxlen=memory_size)
        self.model = self.create_model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, eps=1e-7)
        print("Agent initialized")        
    

    def create_model(self) -> torch.nn.Module:
        model = nn.Sequential(
            nn.Conv2d(in_channels=self.frame_stack_num, out_channels=6, kernel_size=7, stride=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(432, 216),
            nn.ReLU(),
            nn.Linear(216, len(self.action_space))
        )
        return model


    def get_action(self, state_stack):
        with torch.no_grad():
            state_tensor = torch.tensor(state_stack, dtype=torch.float32, device=self.device)
            state_tensor = torch.unsqueeze(state_tensor, 0)
            q_values = self.model(state_tensor)
            action = torch.argmax(q_values).item()
            return action


    def get_action_explore(self, state_stack):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)
        return self.get_action(state_stack)


    def add_to_memory(self, frame_stack, action, reward, next_frame_stack, terminated):
        self.memory.append((frame_stack, action, reward, next_frame_stack, terminated))

    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    

    def replay(self):
        mini_batch = random.sample(self.memory, self.training_batch_size)
        
        states = torch.tensor(np.stack([transition[0] for transition in mini_batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([transition[1] for transition in mini_batch], device=self.device)
        rewards = torch.tensor([transition[2] for transition in mini_batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.stack([transition[3] for transition in mini_batch]), dtype=torch.float32, device=self.device)
        terminated = torch.tensor([transition[4] for transition in mini_batch], device=self.device)

        # Compute Q-values for the current states
        q_values = self.model(states)

        # Compute Q-values for the next states
        with torch.no_grad():
            next_q_values = self.model(next_states)

        # Create a mask to zero out Q-values for terminated states
        mask = torch.logical_not(terminated)

        # Compute target Q-values
        target_q_values = q_values.clone()
        target_q_values[range(self.training_batch_size), actions] = rewards + mask * self.discount_factor * torch.max(next_q_values, dim=1).values

        # Compute the loss
        criterion = nn.MSELoss()
        loss = criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()  


if __name__ == "__main__":
    agent = agent()
    print(agent.model)
