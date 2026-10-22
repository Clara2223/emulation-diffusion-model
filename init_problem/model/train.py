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
    validation_loader=None,
    return_validation_history=False,
):
    """Train by fixed optimizer steps or full loader passes and optionally return epoch losses."""
    if return_validation_history and (epochs is None or validation_loader is None):
        raise ValueError("Validation history requires epochs and a validation data loader")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)     
    model.train()

    def loss_for(z, sigma, epsilon):
        noisy = z + sigma[:, None] * epsilon
        predicted_score = model(noisy, sigma)
        target_score = -epsilon / sigma[:, None]
        weighted_error = (predicted_score - target_score) * sigma[:, None]
        return (weighted_error ** 2).sum(dim=1).mean()

    def batch_loss(batch):
        (z,) = batch
        z = (z.to(DEVICE) - mean) / std
        sigma = torch.exp(torch.empty(z.shape[0], device=DEVICE).uniform_(
            np.log(sigma_min), np.log(sigma_max)))
        epsilon = torch.randn_like(z)
        return loss_for(z, sigma, epsilon)

    history = []
    validation_history = []
    fixed_validation_batches = []
    if validation_loader is not None:
        generator = torch.Generator(device=DEVICE).manual_seed(0)
        for batch in validation_loader:
            z = (batch[0].to(DEVICE) - mean) / std
            sigma = torch.exp(torch.empty(z.shape[0], device=DEVICE).uniform_(
                np.log(sigma_min), np.log(sigma_max), generator=generator))
            epsilon = torch.randn(z.shape, device=DEVICE, generator=generator)
            fixed_validation_batches.append((z, sigma, epsilon))
        if not fixed_validation_batches:
            raise ValueError("Cannot validate with an empty data loader")
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
            if validation_loader is not None:
                model.eval()
                validation_loss, validation_count = 0.0, 0
                with torch.no_grad():
                    for z, sigma, epsilon in fixed_validation_batches:
                        loss = loss_for(z, sigma, epsilon)
                        batch_size = z.shape[0]
                        validation_loss += loss.item() * batch_size
                        validation_count += batch_size
                validation_history.append(validation_loss / validation_count)
                model.train()
    if return_validation_history:
        return model, history, validation_history
    return (model, history) if return_history else model


def train_flow(
    model,
    loader,
    mean,
    std,
    steps=2_000,
    *,
    epochs=None,
    return_history=False,
    validation_loader=None,
    return_validation_history=False,
):
    """Fit flow matching by fixed steps or full loader passes."""
    if return_validation_history and (epochs is None or validation_loader is None):
        raise ValueError("Validation history requires epochs and a validation data loader")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    def loss_for(z, epsilon, t):
        x_t = (1 - t[:, None]) * epsilon + t[:, None] * z
        target_velocity = z - epsilon
        predicted_velocity = model(x_t, t)
        return ((predicted_velocity - target_velocity) ** 2).sum(dim=1).mean()

    def batch_loss(batch):
        (z,) = batch
        z = (z.to(DEVICE) - mean) / std
        epsilon = torch.randn_like(z)
        t = torch.rand(z.shape[0], device=DEVICE)
        return loss_for(z, epsilon, t)

    history = []
    validation_history = []
    fixed_validation_batches = []
    if validation_loader is not None:
        generator = torch.Generator(device=DEVICE).manual_seed(0)
        for batch in validation_loader:
            z = (batch[0].to(DEVICE) - mean) / std
            epsilon = torch.randn(z.shape, device=DEVICE, generator=generator)
            t = torch.rand(z.shape[0], device=DEVICE, generator=generator)
            fixed_validation_batches.append((z, epsilon, t))
        if not fixed_validation_batches:
            raise ValueError("Cannot validate with an empty data loader")
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
            if validation_loader is not None:
                model.eval()
                validation_loss, validation_count = 0.0, 0
                with torch.no_grad():
                    for z, epsilon, t in fixed_validation_batches:
                        loss = loss_for(z, epsilon, t)
                        batch_size = z.shape[0]
                        validation_loss += loss.item() * batch_size
                        validation_count += batch_size
                validation_history.append(validation_loss / validation_count)
                model.train()
    if return_validation_history:
        return model, history, validation_history
    return (model, history) if return_history else model
