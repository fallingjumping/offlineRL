from env import env
from replaybuffer import Replay_buffer
from load_data import historical_dataset
from TD3 import TD3_BC
import torch

max_timesteps = 1e6
filename = f''


device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

myenv = env()
statedim = myenv.state_dim
actiondim = myenv.action_dim
maxaction = torch.tensor(myenv.max_action).to(device)
datasets = historical_dataset()
buffer = Replay_buffer(statedim,actiondim)
policy = TD3_BC(statedim, actiondim, maxaction)
buffer.load_data(datasets)
mean, std = buffer.normalize_state()
# print(mean, std)
for i in range(int(max_timesteps)):
    # print(mean, std)
    policy.train(buffer)
    break


eval_episodes = 10