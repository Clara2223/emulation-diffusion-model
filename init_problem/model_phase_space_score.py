"""Run the current equilibrium phase-space score-model pipeline."""

from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from model.data import load_equilibrium_bundle
from model.model import DEVICE, PhaseSpaceScoreNet
from model.predict import (
    calculate_einstein_diffusion,
    calculate_kramers_rate,
    log_prediction,
    plot_results,
    sample_score,
)
from model.train import train_score


DATA_PATH = Path(__file__).parent / "train_data" / (
    "double_well_data_100000000steps_50000burn_20260918_115938.npz"
)

#/Users/clarawimmelmann/Desktop/Energy_specialCourse/init_problem/train_data/periodic_double_well_chain_100000steps_5000burn.npz

epochs=21
batch_size=512
total_samples=50_0000
steps_per_epoch=total_samples/batch_size   
total_steps = epochs * steps_per_epoch  #20k ish
#-------------------------------------
TRAIN_STEPS = 200_000
MAX_TRAINING_SAMPLES = 100_0000
GENERATED_SAMPLES = 20_000


# Load MD data.
train_data, true_samples, positions, _ = load_equilibrium_bundle(
    DATA_PATH,
    max_samples=MAX_TRAINING_SAMPLES,
    diffusion_frames=50_000,
)

# Standardize the 5D state (x, y, vx, vy, and energy).
mean = train_data.mean(dim=0).to(DEVICE)
std = train_data.std(dim=0).clamp_min(1e-6).to(DEVICE)
loader = DataLoader(TensorDataset(train_data), batch_size=512, shuffle=True)

# Create and train score model.
#-------------------------------------
# - Target: The model learns the Score Function S(x) = ∇_x log p(x), where p(x)
#   is the 5D equilibrium Boltzmann distribution p(x) ∝ exp(-U(x) / k_B*T).
# - Key Insight: Taking the log gradient eliminates the intractable partition
#   function Z (∇_x log Z = 0). The network outputs a 5D vector pointing uphill
#   toward higher probability density (lower potential energy).
model = PhaseSpaceScoreNet().to(DEVICE)
model = train_score(model, loader, mean, std, steps=TRAIN_STEPS)

# Generate equilibrium predictions with 'annealed' Langevin sampling
# ------------------------------------------------------------------------------
# - Langevin Dynamics updates states via:
#   x_{k+1} = x_k + (ε/2) * [Score Vector] + sqrt(ε) * [Gaussian Noise]
#   This balances deterministic drift into energy wells with thermal kicks in discrete level steps of size ε.
generated = sample_score(
    model,
    mean,
    std,
    n_samples=GENERATED_SAMPLES,
    n_levels=12,
    steps_per_level=6,
)

# Calculate the prediction diagnostics and save the comparison plot.
einstein_diffusion = calculate_einstein_diffusion(positions)
kramers_rate, barrier_height = calculate_kramers_rate()
output_path = Path(__file__).parent / "predictions" / (
    f"phase_space_score_{datetime.now():%Y%m%d_%H%M%S}.png"
)
output_path.parent.mkdir(exist_ok=True)
plot_results(true_samples, generated, output_path)
log_path = log_prediction(
    output_path,
    model_version="equilibrium_score_v1_small",
    einstein_diffusion=einstein_diffusion,
    barrier_height=barrier_height,
    kramers_rate=kramers_rate,
)

print(f"Device: {DEVICE}")
print(f"Einstein diffusion coefficient: {einstein_diffusion:.5f}")
print(f"Kramers barrier height: {barrier_height:.5f}")
print(f"Kramers transition rate: {kramers_rate:.5f}")
print(f"Visualization saved as '{output_path}'")
print(f"Prediction log saved as '{log_path}'")
