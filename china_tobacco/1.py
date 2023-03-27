import numpy as np
import torch
ind = np.random.randint(0, 10, size=5)
print(ind)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state = np.array([1,2,3,4,5,6,7,8,9,0,11,22,33,44])
a = torch.FloatTensor(state[ind]).to(device)
print(a)