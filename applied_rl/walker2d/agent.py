import random
import numpy as np
import os
import torch
from torch import nn
from torch.distributions import Normal


class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size))#.to(device)

        self.sigma = torch.nn.Parameter(torch.as_tensor(np.ones(action_size, dtype=np.float32) * 1.5))#.to(device)
        self.train()
        
    def compute_proba(self, state, action):
        mu = self.model(state)
        sigma = self.sigma
        distrib = Normal(mu, sigma) # batch_size x action_size
        return torch.exp(distrib.log_prob(action).sum(-1)), distrib

    def act(self, state):
        mu = self.model(state)
        sigma = self.sigma
        distrib = Normal(mu, sigma)
        action = distrib.sample()
        return torch.tanh(action), action, distrib


class Agent:
    def __init__(self):
        self.agent = Actor(22, 6) # [state_size, action_size, seed]
        self.agent.load_state_dict(torch.load(__file__[:-8] + "/1450.96_125.14_agent.pth"))
        
    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array(state)).float()
            action_values = self.agent.act(state)
            # return np.argmax(action_values[1].cpu().data.numpy())
            return action_values[0].cpu().data

    def reset(self):
        pass

