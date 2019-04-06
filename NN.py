import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, actions, input_size, hidden_units=512):
        super(QNetwork, self).__init__()
        self.actions = actions
        self.FC = nn.Sequential(
            nn.Linear(input_size, hidden_units),
            nn.LeakyReLU(),
            nn.Linear(hidden_units, actions)
        )
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, features, next_features):
        return self.FC(features)
