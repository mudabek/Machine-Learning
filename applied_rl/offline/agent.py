import random
import numpy as np
import os
import torch
from torch import nn


class BehavioralCloning(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(19, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, 5),
            nn.Tanh()
        )

    def get_action(self, state):
        return self.model(state)


class Agent:
    def __init__(self):
        self.agent = BehavioralCloning(256)
        self.agent.load_state_dict(torch.load(__file__[:-8] + "/agent.pth"))
        self.agent = self.agent.double()
        
    def act(self, state):
        state = torch.tensor(np.array(state)).double()
        return self.agent.get_action(state)

    def reset(self):
        pass

