import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, enable_dueling_dqn=True):
        super(DQN, self).__init__()

        self.enable_dueling_dqn = enable_dueling_dqn

        self.fc1 = nn.Linear(state_dim, hidden_dim)

        if self.enable_dueling_dqn:
            # Value stream
            self.fc_value = nn.Linear(hidden_dim, 256)
            self.value = nn.Linear(256, 1)
            # Advantage stream
            self.fc_advantage = nn.Linear(hidden_dim, 256)
            self.advantages = nn.Linear(256, action_dim)

        else:
            self.output = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        if self.enable_dueling_dqn:
            # Value calculation
            v = self.fc_value(x)
            v = F.relu(v)  
            V = self.value(v)
            # Advantage calculation
            a = self.fc_advantage(x)
            a = F.relu(a)
            A = self.advantages(a)
            # Q calculation
            Q = V + A - torch.mean(A, dim=1, keepdim=True)
            
        else:
            Q = self.output(x)
        return Q
    