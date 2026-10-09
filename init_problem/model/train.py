import numpy as np
import torch
from .diffusion_model import DEVICE

def train_score(model, loader, mean, std, steps=2_000, sigma_min=0.02, sigma_max=1.0):
    """denoising score matching at noise levels (defined as discrete levels with a no. 
    of steps per level)"""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)     
    iterator = iter(loader)
    model.train()
    
    #next() returns the next item from the iterator 
    # Here: it retrieves a training data batch from the DataLoader
    for _ in range(steps):
        try:
            (z,) = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            (z,) = next(iterator)
            
        z = (z.to(DEVICE) - mean) / std
        
        #sigma: a sampled random noise level for each trajectory in training batch
        # uses a geometric (scales exp, log-uniform) distribution
        sigma = torch.exp(torch.empty(z.shape[0], device=DEVICE).uniform_(
                np.log(sigma_min), np.log(sigma_max)))
        
        epsilon = torch.randn_like(z)
        x = z + sigma[:, None] * epsilon
        predicted_score = model(x, sigma)
        # Conditional score: grad_x log p(x|z) = -epsilon / sigma
        target_score = -epsilon / sigma[:, None]
        # Averaging this regression over z and noise: network learn
        # the marginal score grad_x log p(x), a vector field over x
        loss = ((predicted_score - target_score) ** 2).sum(dim=1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def train_flow(model, loader, mean, std, steps=2_000):
    """Fit straight-line flow matching from Gaussian noise to data."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    iterator = iter(loader)
    model.train()

    for _ in range(steps):
        try:
            (z,) = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            (z,) = next(iterator)

        z = (z.to(DEVICE) - mean) / std
        epsilon = torch.randn_like(z)
        t = torch.rand(z.shape[0], device=DEVICE)
        x_t = (1 - t[:, None]) * epsilon + t[:, None] * z
        
        # Conditional vector field (1 velocity) along this specific noise-to-data path
        target_velocity = z - epsilon
        predicted_velocity = model(x_t, t)
        # 'Regression over paths' learns the marginal velocity field at (x_t, t)
        loss = ((predicted_velocity - target_velocity) ** 2).sum(dim=1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model
