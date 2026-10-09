import numpy as np
import torch
from .diffusion_model import DEVICE

def train_score(
    model,
    loader,
    mean,
    std,
    steps=2_000,
    sigma_min=0.02,
    sigma_max=1.0,
    *,
    epochs=None,
    return_history=False,
):
    """Train by fixed optimizer steps or full loader passes and optionally return epoch losses."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)     
    model.train()

    def batch_loss(batch):
        (z,) = batch
        z = (z.to(DEVICE) - mean) / std
        sigma = torch.exp(torch.empty(z.shape[0], device=DEVICE).uniform_(
            np.log(sigma_min), np.log(sigma_max)))
        epsilon = torch.randn_like(z)
        noisy = z + sigma[:, None] * epsilon
        predicted_score = model(noisy, sigma)
        target_score = -epsilon / sigma[:, None]
        return ((predicted_score - target_score) ** 2).sum(dim=1).mean()

    history = []
    if epochs is None:
        iterator = iter(loader)
        for _ in range(steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            loss = batch_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    else:
        for _ in range(epochs):
            epoch_loss, sample_count = 0.0, 0
            for batch in loader:
                loss = batch_loss(batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_size = batch[0].shape[0]
                epoch_loss += loss.detach().item() * batch_size
                sample_count += batch_size
            if sample_count == 0:
                raise ValueError("Cannot train with an empty data loader")
            history.append(epoch_loss / sample_count)
    return (model, history) if return_history else model


def train_flow(model, loader, mean, std, steps=2_000, *, epochs=None, return_history=False):
    """Fit flow matching by fixed steps or full loader passes."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    def batch_loss(batch):
        (z,) = batch
        z = (z.to(DEVICE) - mean) / std
        epsilon = torch.randn_like(z)
        t = torch.rand(z.shape[0], device=DEVICE)
        x_t = (1 - t[:, None]) * epsilon + t[:, None] * z
        target_velocity = z - epsilon
        predicted_velocity = model(x_t, t)
        return ((predicted_velocity - target_velocity) ** 2).sum(dim=1).mean()

    history = []
    if epochs is None:
        iterator = iter(loader)
        for _ in range(steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            loss = batch_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    else:
        for _ in range(epochs):
            epoch_loss, sample_count = 0.0, 0
            for batch in loader:
                loss = batch_loss(batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_size = batch[0].shape[0]
                epoch_loss += loss.detach().item() * batch_size
                sample_count += batch_size
            if sample_count == 0:
                raise ValueError("Cannot train with an empty data loader")
            history.append(epoch_loss / sample_count)
    return (model, history) if return_history else model
