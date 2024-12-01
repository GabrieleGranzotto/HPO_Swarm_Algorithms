import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FeedForwardNN(nn.Module):

    def __init__(self, in_dim, out_dim):
        
        super(FeedForwardNN, self).__init__()

        self.layer1 = nn.Linear(in_dim, 8)         # first layer of the NN [in_dim, 64]
        self.layer2 = nn.Linear(8, 8)             # second layer of the NN [64, 128]
        self.layer3 = nn.Linear(8, out_dim)        # third and last layer of the NN [128, out_dim]

    def forward(self, obs):
        
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype = torch.float)

        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)

        return output