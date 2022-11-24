import torch
import torch.nn as nn


class DQN_Model(nn.Module):
    def __init__(self, input_size, n_actions: int):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, n_actions)
        )

    def forward(self, env_input):
        x = torch.flatten(env_input, start_dim=1)
        x = self.nn(x)
        x = self.classifier(x)
        return x
