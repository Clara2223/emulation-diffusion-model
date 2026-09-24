import numpy as np
import torch

from .model import DEVICE


def train_score(model, loader, mean, std, steps=2_000, sigma_min=0.02, sigma_max=1.0):
    """Train with denoising score matching at several noise levels. (defined as discrete levels with a no. of steps per level)"""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)     #experiment with lr
    iterator = iter(loader)
    model.train()
    
    for _ in range(steps):
        try:
            (clean,) = next(iterator)           #next() returns the next item from the iterator. Here: it retrieves a training data batch from the DataLoader.
        except StopIteration:
            iterator = iter(loader)
            (clean,) = next(iterator)
            
        clean = (clean.to(DEVICE) - mean) / std     #clean=one training data batch
        
        #sigma: a sampled random noise level for each trajectory in training batch. uses a geometric (scales exp) (log-uniform) distribution.
        sigma = torch.exp(
            torch.empty(clean.shape[0], device=DEVICE).uniform_(
                np.log(sigma_min), np.log(sigma_max)))
        
        noise = torch.randn_like(clean)
        prediction = model(clean + sigma[:, None] * noise, sigma)
        # i didn't 100% understand this part, but it seems to be the negative of the noise divided by the sigma (noise level). 
        # the score vector at that noise level
        target = -noise / sigma[:, None]
        
        loss = ((prediction - target) ** 2).sum(dim=1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model
