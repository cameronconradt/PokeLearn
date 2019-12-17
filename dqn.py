import torch
import random
import torch.nn as nn
import numpy as np


# Q-Value Network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        hidden_size = 2000

        self.net = nn.Sequential(nn.Linear(state_size, hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, action_size))

    def forward(self, x):
        """Estimate q-values given state

          Args:
              state (tensor): current state, size (batch x state_size)

          Returns:
              q-values (tensor): estimated q-values, size (batch x action_size)
        """
        return self.net(x)


