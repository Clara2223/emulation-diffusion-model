import argparse
from datetime import datetime
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import load_equilibrium_bundle
from .diffusion_model import DEVICE, DiffusionModel, MODEL_VERSION as DIFFUSION_VERSION
from .flow_model import FlowModel, MODEL_VERSION as FLOW_VERSION
from .predict import (
    calculate_einstein_diffusion,
    calculate_kramers_rate,
    log_prediction,
    plot_results,
    sample_flow,
    sample_score,
)
from .train import train_flow, train_score

#DEFAULT_DATA = Path(__file__).resolve().parents[1] / "train_data" / "2D_double_well_data_100000000steps_50000burn.npz"
DEFAULT_DATA = Path(__file__).resolve().parents[1] / "train_data" / "3D_double_well_data_100000000steps_50000burn.npz"

#DEFAULT_DATA = Path(__file__).resolve().parents[1] / "train_data" / "2D_periodic_double_well_chain_1e8steps_5e4burn_20261009_0930.npz",
#DEFAULT_DATA = Path(__file__).resolve().parents[1] / "train_data" / "2D_double_well_chain_1e8steps_5e4burn_20261009_0930.npz",
    

#"double_well_data_100000000steps_50000burn_20260918_115938.npz"

def main():
    #inputs 
    parser = argparse.ArgumentParser(description="Train a 2D position diffusion or flow model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--train-steps", type=int, default=2_000)
    parser.add_argument("--max-samples", type=int, default=50_000)
    parser.add_argument("--generated-samples", type=int, default=1_000)
    parser.add_argument("--model", choices=("diffusion", "flow"), default="diffusion")
    parser.add_argument("--n-levels", type=int, default=12)
    parser.add_argument("--steps-per-level", type=int, default=6)
    parser.add_argument("--ode-steps", type=int, default=100)
    parser.add_argument("--diffusion-frames", type=int, default=50_000)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    #data handling: load data train and compute mean/std for normalization
    train_data, true_samples, positions = load_equilibrium_bundle(
        args.data, args.max_samples, args.diffusion_frames)
    
    mean = train_data.mean(dim=0).to(DEVICE)
    std = train_data.std(dim=0).clamp_min(1e-6).to(DEVICE)
    train_loader = DataLoader(TensorDataset(train_data), batch_size=512, shuffle=True)
    
    #choose model type and train, generate samples
    if args.model == "flow":
        model = train_flow(FlowModel().to(DEVICE), train_loader, mean, std, args.train_steps)
        generated = sample_flow(model, mean, std, args.generated_samples, args.ode_steps)
        model_version = FLOW_VERSION
    else:
        model = train_score(DiffusionModel().to(DEVICE), train_loader, mean, std, args.train_steps)
        generated = sample_score(model, mean, std, args.generated_samples, args.n_levels, args.steps_per_level)
        model_version = DIFFUSION_VERSION
    

    #results: save plot, 'stats' and log file
    einstein = calculate_einstein_diffusion(positions)
    kramers_rate, barrier_height = calculate_kramers_rate()
    output = (args.data.parent/ ".."/ "predictions"/ f"xy_{args.model}_{datetime.now():%Y%m%d_%H%M}.png")
    output.parent.mkdir(exist_ok=True)
    plot_results(true_samples, generated, output, show=args.show)
    log_path = log_prediction(
        output,
        model_version=model_version,
        einstein_diffusion=einstein,
        barrier_height=barrier_height,
        kramers_rate=kramers_rate,
    )
    print(f"Device: {DEVICE}")
    print(f"Einstein diffusion coefficient: {einstein:.5f}")
    print(f"Kramers barrier height: {barrier_height:.5f}")
    print(f"Kramers transition rate: {kramers_rate:.5f}")
    print(f"Visualization saved as '{output.resolve()}'")
    print(f"Prediction log saved as '{log_path.resolve()}'")


if __name__ == "__main__":
    main()
