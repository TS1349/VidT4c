import torch.nn as nn
import torch.nn.functional as F


class MeanPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states.mean(1)
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    

class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class MatchingHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.pooler = Pooler(hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.fc(self.pooler(x))
        return x