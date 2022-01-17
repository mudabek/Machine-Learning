import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import tqdm


class TransitionDataset(Dataset):
    def __getitem__(self, index):
        state, action, next_state, reward, done = self.transitions[index]
        return torch.tensor(state, dtype=torch.float32), torch.tensor(action, dtype=torch.float32), \
               torch.tensor(next_state, dtype=torch.float32), torch.tensor([reward], dtype=torch.float32), \
               torch.tensor([done], dtype=torch.float32)

    def __init__(self, path):
        self.transitions = np.load(path, allow_pickle=True)["arr_0"]

    def __len__(self):
        return len(self.transitions)


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

    def save(self, info):
        torch.save(self.actor.state_dict(), info + "_agent.pth")


def train():
    model = BehavioralCloning(256)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    dataset = TransitionDataset("optimal.npz")
    for _ in tqdm.tqdm(range(2000)):
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        for batch in dataloader:
            state, action, _, _, _ = batch
            action_pred = model.get_action(state)
            loss = F.mse_loss(action_pred, action)
            optim.zero_grad()
            loss.backward()
            optim.step()
    return model


if __name__=="__main__":
    model = train()