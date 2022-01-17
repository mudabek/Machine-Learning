import random
import numpy as np
import os
import torch
from torch import nn

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        
    def forward(self, state):
        return self.model(state)

class Agent:
    def __init__(self):
        self.agent = Actor(28, 8) # [state_size, action_size]
        self.agent.load_state_dict(torch.load(__file__[:-8] + "/c.pth"))
        
    def act(self, state):
        state = torch.tensor(np.array(state))
        return self.agent(state)

    def reset(self):
        pass

