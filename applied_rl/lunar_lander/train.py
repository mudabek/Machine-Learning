from gym import make
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from collections import deque, namedtuple
import random


GAMMA = 0.99
INITIAL_STEPS = 1024
TRANSITIONS = 1000000
STEPS_PER_UPDATE = 4
LEARNING_RATE = 5e-4
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):

    def __init__(self, state_size, action_size):
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


class ReplayBuffer:

    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)


    def add(self, e):
        self.memory.append(self.experience(e[0], e[1], e[2], e[3], e[4]))
    

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)


    def __len__(self):
        return len(self.memory)


class DQN():

    def __init__(self, state_dim, action_dim, seed):
        self.steps = 0 # Do not change
        self.memory = ReplayBuffer(action_dim, BUFFER_SIZE, BATCH_SIZE, seed) # experience replay buffer
        self.state_size = state_dim
        self.action_size = action_dim
        self.seed = random.seed(seed)

        # Model for training
        self.model = QNetwork(state_dim, action_dim, seed).to(device)
        self.model_local = QNetwork(state_dim, action_dim, seed).to(device)
        self.optimizer = Adam(self.model_local.parameters(), lr=LEARNING_RATE)


    def consume_transition(self, transition):
        # Add transition to a replay buffer
        self.memory.add(transition)


    def sample_batch(self):
        # Sample batch from a replay buffer
        return self.memory.sample()
        

    def train_step(self, batch):
        # Use batch to update DQN's network.
        states, actions, next_states, rewards, dones = batch
        ## Compute and minimize the loss
        ### Extract next maximum estimated value from target network
        q_targets_next = self.model(next_states).detach().max(1)[0].unsqueeze(1)
        ### Calculate target value from bellman equation
        q_targets = rewards + GAMMA * q_targets_next * (1 - dones)
        ### Calculate expected value from local network
        q_expected = self.model_local(states).gather(1, actions)
        
        ### Loss calculation (we used Mean squared error)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_target_network()
        

    def update_target_network(self, tau=1e-3):
        # Update weights of a target Q-network
        for target_param, local_param in zip(self.model.parameters(), self.model_local.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


    def act(self, state):
        # Compute an action
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.model_local.eval()
        with torch.no_grad():
            action_values = self.model_local(state)
        self.model_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))


    def update(self, transition):
        self.consume_transition(transition)
        if self.steps % STEPS_PER_UPDATE == 0:
            batch = self.sample_batch()
            self.train_step(batch)
            self.update_target_network()
        self.steps += 1


    def save(self):
        torch.save(self.model.state_dict(), "agent.pth")


def evaluate_policy(agent, episodes=5):
    env = make("LunarLander-v2")
    returns = []

    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.1
        
        while not done:
            state, reward, done, _ = env.step(agent.act(state))
            total_reward += reward
        returns.append(total_reward)

    return returns


if __name__ == "__main__":

    env = make("LunarLander-v2")
    dqn = DQN(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n, seed=123)
    eps = 0.01
    state = env.reset()
    
    for _ in range(INITIAL_STEPS):
        action = env.action_space.sample()

        next_state, reward, done, _ = env.step(action)
        dqn.consume_transition((state, action, next_state, reward, done))
        
        state = next_state if not done else env.reset()
        
    
    for i in range(TRANSITIONS):
        #Epsilon-greedy policy
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = dqn.act(state)

        next_state, reward, done, _ = env.step(action)
        dqn.update((state, action, next_state, reward, done))
        
        state = next_state if not done else env.reset()
        
        if (i + 1) % (TRANSITIONS//100) == 0:
            rewards = evaluate_policy(dqn, 5)
            print(f"Step: {i+1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}")
            dqn.save()