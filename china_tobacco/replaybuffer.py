import numpy as np
import torch


class Replay_buffer:
    def __init__(self, statedim, actiondim) -> None:
        self.statedim = statedim
        self.actiondim = actiondim

        self.state = np.zeros(shape=(self.statedim,), dtype=np.float32)
        self.action = np.zeros(shape=(self.actiondim,), dtype=np.float32)
        self.next_state = np.zeros(shape=(self.statedim,), dtype=np.float32)
        self.reward = np.zeros(shape=(1,), dtype=np.float32)
        self.done = np.zeros(shape=(1,), dtype=np.float32)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def load_data(self, dataset):
        self.state = dataset['states']
        self.action = dataset['actions']
        self.next_state = dataset['next_states']
        self.reward = dataset['rewards']
        self.done = dataset['terminals']

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.state)-1, batch_size)
        return (
            torch.tensor(self.state[ind], dtype=torch.float32).to(self.device),
            torch.tensor(self.action[ind], dtype=torch.float32).to(self.device),
            torch.tensor(self.next_state[ind], dtype=torch.float32).to(self.device),
            torch.tensor(self.reward[ind], dtype=torch.float32).to(self.device),
            torch.tensor(self.done[ind]).to(self.device)
        )

    def normalize_state(self, eps=1e-3):
        mean = self.state.mean(0, keepdims=True)
        std = self.state.std(0, keepdims=True) + eps
        self.state = (self.state - mean) / std
        self.next_state = (self.next_state - mean) / std
        return mean, std
