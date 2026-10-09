import torch
import torch.nn as nn

from .data import STATE_DIM


MODEL_VERSION = "xy_flow_v1"


class FlowModel(nn.Module):
    """Predict a flow velocity: time is transport coord (& not MD time)"""

    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, STATE_DIM),
        )

    def forward(self, state, time):
        inputs = torch.cat([state, time[:, None]], dim=1)
        return self.net(inputs)