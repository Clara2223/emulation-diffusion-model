import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Diffusion3D(nn.Module):
    """Noise-conditioned score model for 3D samples or conditional transitions."""

    def __init__(self, state_dim=3, condition_dim=0, hidden_dim=64, embedding_dim=16):
        super().__init__()
        self.state_dim = state_dim
        self.condition_dim = condition_dim
        frequencies = torch.exp(torch.linspace(np.log(1.0), np.log(1000.0), embedding_dim // 2))
        self.register_buffer("frequencies", frequencies)
        self.net = nn.Sequential(
            nn.Linear(state_dim + condition_dim + embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, state, sigma, condition=None):
        log_sigma = torch.log(sigma.clamp_min(1e-6)).unsqueeze(1)
        angles = log_sigma * self.frequencies.unsqueeze(0)
        features = [state]
        if self.condition_dim:
            if condition is None or condition.shape[1] != self.condition_dim:
                raise ValueError(f"Expected a condition with {self.condition_dim} features")
            features.append(condition)
        features.append(torch.cat((torch.sin(angles), torch.cos(angles)), dim=1))
        return self.net(torch.cat(features, dim=1))


class Flow3D(nn.Module):
    """Time-conditioned flow model for 3D samples or conditional transitions."""

    def __init__(self, state_dim=3, condition_dim=0, hidden_dim=64):
        super().__init__()
        self.state_dim = state_dim
        self.condition_dim = condition_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim + condition_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, state, time, condition=None):
        features = [state, time[:, None]]
        if self.condition_dim:
            if condition is None or condition.shape[1] != self.condition_dim:
                raise ValueError(f"Expected a condition with {self.condition_dim} features")
            features.append(condition)
        return self.net(torch.cat(features, dim=1))


def _fit(model, states, conditions, epochs, batch_size, loss_fn, validation):
    if len(states) == 0:
        raise ValueError("Cannot train on an empty dataset")
    if conditions is not None and len(conditions) != len(states):
        raise ValueError("Condition and target counts must match")
    tensors = (states,) if conditions is None else (states, conditions)
    loader = DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []
    validation_history = []

    for _ in range(epochs):
        model.train()
        total_loss, count = 0.0, 0
        for batch in loader:
            state = batch[0].to(DEVICE)
            condition = batch[1].to(DEVICE) if conditions is not None else None
            loss = loss_fn(state, condition)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item() * len(state)
            count += len(state)
        history.append(total_loss / count)
        if validation is not None:
            val_states, val_conditions = validation
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for start in range(0, len(val_states), batch_size):
                    state = val_states[start:start + batch_size].to(DEVICE)
                    condition = (
                        None if val_conditions is None
                        else val_conditions[start:start + batch_size].to(DEVICE)
                    )
                    val_loss += loss_fn(state, condition).item() * len(state)
            validation_history.append(val_loss / len(val_states))
    return model, history, validation_history


def train_diffusion(
    model, states, conditions=None, epochs=100, batch_size=512, validation=None,
):
    def loss_fn(state, condition):
        sigma = torch.exp(torch.empty(len(state), device=DEVICE).uniform_(np.log(0.02), 0.0))
        epsilon = torch.randn_like(state)
        noisy = state + sigma[:, None] * epsilon
        score = model(noisy, sigma, condition)
        return (((score + epsilon / sigma[:, None]) * sigma[:, None]) ** 2).sum(dim=1).mean()

    return _fit(model, states, conditions, epochs, batch_size, loss_fn, validation)


def train_flow(model, states, conditions=None, epochs=100, batch_size=512, validation=None):
    def loss_fn(state, condition):
        epsilon = torch.randn_like(state)
        time = torch.rand(len(state), device=DEVICE)
        interpolated = (1 - time[:, None]) * epsilon + time[:, None] * state
        target_velocity = state - epsilon
        velocity = model(interpolated, time, condition)
        return ((velocity - target_velocity) ** 2).sum(dim=1).mean()

    return _fit(model, states, conditions, epochs, batch_size, loss_fn, validation)


def sample_diffusion(model, n_samples, condition=None, n_levels=12, steps_per_level=50):
    model.eval()
    state = torch.randn(n_samples, model.state_dim, device=DEVICE)
    sigmas = torch.exp(torch.linspace(np.log(1.0), np.log(0.02), n_levels, device=DEVICE))
    with torch.no_grad():
        for sigma in sigmas:
            sigma_batch = sigma.expand(n_samples)
            step_size = 0.02 * sigma**2
            for _ in range(steps_per_level):
                state = state + step_size * model(state, sigma_batch, condition)
                state = state + torch.sqrt(2 * step_size) * torch.randn_like(state)
    return state


def sample_flow(model, n_samples, condition=None, ode_steps=50):
    model.eval()
    state = torch.randn(n_samples, model.state_dim, device=DEVICE)
    step_size = 1.0 / ode_steps
    with torch.no_grad():
        for step in range(ode_steps):
            time = torch.full((n_samples,), step / ode_steps, device=DEVICE)
            state = state + step_size * model(state, time, condition)
    return state
