import torch
import torch.nn as nn

from .data import STATE_DIM

DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")
MODEL_VERSION = "xy_diffusion_v1"


class DiffusionModel(nn.Module):
    """Predict the score of a noisy two-dimensional position."""

    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, STATE_DIM),
        )

    def forward(self, state, noise_level):
        inputs = torch.cat([state, noise_level[:, None]], dim=1)
        return self.net(inputs)