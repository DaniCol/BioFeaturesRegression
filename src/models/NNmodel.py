import torch
import torch.nn as nn


class BIONNregressor(nn.Module):
    def __init__(self, num_features):
        super(BIONNregressor, self).__init__()
        # Define layers
        self.linear = nn.Sequential(
            nn.Linear(num_features, 16),
            nn.Linear(16, 32),
            nn.Linear(32, 16),
            nn.Linear(16, 1),
        )

        # Activation layer
        self.sig = nn.Sigmoid()

    def forward(self, x):
        output = self.linear(x)
        output = self.sig(output)
        return output
