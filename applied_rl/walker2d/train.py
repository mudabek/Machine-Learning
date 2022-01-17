import pybullet_envs
# Don't forget to install PyBullet!
from gym import make
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F
from torch.optim import Adam
import random

ENV_NAME = "Walker2DBulletEnv-v0"

LAMBDA = 0.92
GAMMA = 0.99

ACTOR_LR = 3e-4
CRITIC_LR = 1e-3
LR = 3e-5

CLIP = 0.4
ENTROPY_COEF = 0#1e-2
MAX_GRAD_NORM = 0.5
BATCHES_PER_UPDATE = 64
BATCH_SIZE = 128

MIN_TRANSITIONS_PER_UPDATE = 512
MIN_EPISODES_PER_UPDATE = 4

ITERATIONS = 1500

device = "cpu"#cuda:0"

    
def compute_lambda_returns_and_gae(trajectory):
    lambda_returns = []
    gae = []
    last_lr = 0.
    last_v = 0.
    for _, _, r, _, v in reversed(trajectory):
        ret = r + GAMMA * (last_v * (1 - LAMBDA) + last_lr * LAMBDA)
        last_lr = ret
        last_v = v
        lambda_returns.append(last_lr)
        gae.append(last_lr - v)
    
    # Each transition contains state, action, old action probability, value estimation and advantage estimation
    return [(s, a, p, v, adv) for (s, a, _, p, _), v, adv in zip(trajectory, reversed(lambda_returns), reversed(gae))]
    


class Actor(nn.Module):
    
    def __init__(self, state_size, action_size, hidden_size=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)).to(device)

        self.sigma = torch.nn.Parameter(torch.as_tensor(np.zeros(action_size, dtype=np.float32))).to(device)
        self.train()

    def get_action_distribution(self, state):
        mu = self.model(state)
        sigma = torch.exp(self.sigma)

        return Normal(mu, sigma) 


    def compute_proba(self, state, action):
        # Returns probability of action according to current policy and distribution of actions
        distrib = self.get_action_distribution(state)
        action_logprob = distrib.log_prob(action).sum(-1) # probability of actions [action_size] 

        return action_logprob, distrib
        

    def act(self, state):
        dist = self.get_action_distribution(state)
        action = dist.sample()

        return torch.tanh(action), action, dist
        
        
class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.train()
        
    def forward(self, state):
        return self.model(state).squeeze(-1)


class PPO:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)#.train()
        self.critic = Critic(state_dim)#.train()

        self.optim = Adam(list(self.actor.parameters()) + list(self.critic.parameters()), LR)


    def compute_loss_actor(self, state, action, advantage, old_prob):
        logp, dist = self.actor.compute_proba(state, action)
        old_prob = torch.log(old_prob)
        ratio = torch.exp(logp - old_prob)
        clip_adv = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * advantage
        loss_actor = -(torch.min(ratio * advantage, clip_adv)).mean()

        entropy_loss = -(ENTROPY_COEF * dist.entropy()[0,0])

        return loss_actor + entropy_loss


    def compute_loss_critic(self, state, reward):
        
        return 0.5 * F.mse_loss(reward, self.critic(state))


    def update(self, trajectories):
        transitions = [t for traj in trajectories for t in traj] # Turn a list of trajectories into list of transitions
        state, action, old_prob, target_value, advantage = zip(*transitions)
        state = np.array(state)
        action = np.array(action)
        old_prob = np.array(old_prob)
        target_value = np.array(target_value)
        advantage = np.array(advantage)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        
        
        for _ in range(BATCHES_PER_UPDATE):
            idx = np.random.randint(0, len(transitions), BATCH_SIZE) # Choose random batch
            s = torch.tensor(state[idx]).float().to(device)
            a = torch.tensor(action[idx]).float().to(device)
            op = torch.tensor(old_prob[idx]).float().to(device) # Probability of the action in state s.t. old policy
            v = torch.tensor(target_value[idx]).float().to(device) # Estimated by lambda-returns 
            adv = torch.tensor(advantage[idx]).float().to(device) # Estimated by generalized advantage estimation 
        
            loss_actor = self.compute_loss_actor(s, a, adv, op)
            loss_critic = self.compute_loss_critic(s, v)
            total_loss = loss_actor + loss_critic

            self.optim.zero_grad()
            total_loss.backward()
            # Clip grad norm
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD_NORM)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM)
            self.optim.step()

            
    def get_value(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state])).float().to(device)
            value = self.critic(state)
        return value.cpu().item()


    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state])).float().to(device)
            action, pure_action, distr = self.actor.act(state)
            prob = torch.exp(distr.log_prob(pure_action).sum(-1))
        return action.cpu().numpy()[0], pure_action.cpu().numpy()[0], prob.cpu().item()


    def save(self, info):
        # torch.save(self.actor, "agent.pkl")
        torch.save(self.actor.state_dict(), info + "_agent.pth")


def evaluate_policy(env, agent, episodes=5):
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.
        
        while not done:
            state, reward, done, _ = env.step(agent.act(state)[0])
            total_reward += reward
        returns.append(total_reward)
    return returns
   

def sample_episode(env, agent):
    s = env.reset()
    d = False
    trajectory = []
    while not d:
        a, pa, p = agent.act(s)
        v = agent.get_value(s)
        ns, r, d, _ = env.step(a)
        trajectory.append((s, pa, r, p, v))
        s = ns
    return compute_lambda_returns_and_gae(trajectory)

if __name__ == "__main__":
    env = make(ENV_NAME)
    ppo = PPO(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])
    state = env.reset()
    episodes_sampled = 0
    steps_sampled = 0
    
    for i in range(ITERATIONS):
        trajectories = []
        steps_ctn = 0
        
        while len(trajectories) < MIN_EPISODES_PER_UPDATE or steps_ctn < MIN_TRANSITIONS_PER_UPDATE:
            traj = sample_episode(env, ppo)
            steps_ctn += len(traj)
            trajectories.append(traj)
        episodes_sampled += len(trajectories)
        steps_sampled += steps_ctn

        ppo.update(trajectories)        
        
        if (i + 1) % (ITERATIONS//100) == 0:
            rewards = evaluate_policy(env, ppo, 5)
            print(f"Step: {i+1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}, Episodes: {episodes_sampled}, Steps: {steps_sampled}")
            ppo.save()
