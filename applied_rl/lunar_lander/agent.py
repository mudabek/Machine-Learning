import random
import numpy as np
import os
import torch
from torch import nn
from torch.nn import functional as F

class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)


class Agent():
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.agent = QNetwork(8, 4, 1) # [state_size, action_size, seed]
        self.agent.load_state_dict(torch.load(__file__[:-8] + "/agent.pth"))
        self.agent.to(self.device)

    def act(self, state):
        action_values = self.agent(torch.from_numpy(state).to(self.device))
        return np.argmax(action_values.cpu().data.numpy())