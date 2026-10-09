from pathlib import Path

import numpy as np
import torch


STATE_DIM = 2


def load_equilibrium_data(path: Path, max_samples: int, seed: int = 7):
    """Load a random subset of two-dimensional position samples."""
    with np.load(path) as data:
        positions = data["positions"]
        rng = np.random.default_rng(seed)
        count = min(max_samples, len(positions))
        indices = rng.choice(len(positions), size=count, replace=False)
        samples = positions[indices]
    return torch.from_numpy(samples.astype(np.float32, copy=False))


def load_equilibrium_bundle(path: Path, max_samples: int, diffusion_frames: int, seed: int = 7):
    """Read 2D training samples and position-only reference data."""
    with np.load(path) as data:
        positions = data["positions"]
        rng = np.random.default_rng(seed)
        count = min(max_samples, len(positions))
        indices = rng.choice(len(positions), size=count, replace=False)
        samples = positions[indices]
        true_samples = positions.copy()
        diffusion_positions = positions[:diffusion_frames].copy()
    return (
        torch.from_numpy(samples.astype(np.float32, copy=False)),
        true_samples,
        diffusion_positions,
    )


def load_training_bundle(path: Path, max_samples: int, diffusion_frames: int, seed: int = 7):
    """Read the archive once and prepare training and diagnostic arrays."""
    with np.load(path) as data:
        positions = data["positions"]
        velocities = data["velocities"]
        energies = data["energy"][:, None]
        rng = np.random.default_rng(seed)
        count = min(max_samples, len(positions) - 1)
        indices = rng.choice(len(positions) - 1, size=count, replace=False)
        current = np.concatenate((positions[indices], velocities[indices], energies[indices]), axis=1)
        following = np.concatenate(
            (positions[indices + 1], velocities[indices + 1], energies[indices + 1]), axis=1
        )
        train_data = torch.from_numpy(current.copy())
        true_samples = np.column_stack(
            (positions[:5000], velocities[:5000], energies[:5000, 0])
        )
        diffusion_positions = positions[:diffusion_frames].copy()
        diffusion_velocities = velocities[:diffusion_frames].copy()

    return (
        train_data,
        torch.from_numpy(current.copy()),
        torch.from_numpy(following.copy()),
        true_samples,
        diffusion_positions,
        diffusion_velocities,
    )
