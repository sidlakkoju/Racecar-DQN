import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np 



class agent:

    def __init__(self, frame_stack_num, action_space, learning_rate, memory_size):

        self.action_space = action_space    
        self.frame_stack_num = frame_stack_num
        
        self.memory = deque(maxlen=memory_size)
        self.model = self.create_model()
        self.criterion = nn.MSELoss()
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
        state_stack = state_stack[np.newaxis, :, :, :]
        state_tensor = torch.tensor(state_stack, dtype=torch.float32)
        q_values = self.model(state_tensor)
        action = torch.argmax(q_values).item()
        return action






if __name__ == "__main__":
    agent = agent()
    print(agent.model)
