import argparse
from datetime import datetime
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import get_potential_settings, load_equilibrium_bundle
from .diffusion_model import DEVICE, DiffusionModel, MODEL_VERSION as DIFFUSION_VERSION
from .flow_model import FlowModel, MODEL_VERSION as FLOW_VERSION
from .predict import (
    calculate_einstein_diffusion,
    calculate_kramers_rate,
    histogram_distances,
    log_prediction,
    plot_training_loss,
    plot_results,
    sample_flow,
    sample_score,
)
from .train import train_flow, train_score

#DEFAULT_DATA = Path(__file__).resolve().parents[1] / "train_data" / "2D_double_well_data_100000000steps_50000burn.npz"
#DEFAULT_DATA = Path(__file__).resolve().parents[1] / "train_data" / "3D_double_well_data_100000000steps_50000burn.npz"
#DEFAULT_DATA = Path(__file__).resolve().parents[1] / "train_data" / "2D_periodic_double_well_chain_1e8steps_5e4burn_20261009_0930.npz",
#DEFAULT_DATA = Path(__file__).resolve().parents[1] / "train_data" / "2D_double_well_chain_1e8steps_5e4burn_20261009_0930.npz",
    
### double well
#DEFAULT_DATA = Path(__file__).resolve().parents[1] /"MD"/ "train_data" / "2D_double_well_1e8steps_5e4burn_20261009_1429.npz"
#DEFAULT_DATA = Path(__file__).resolve().parents[1] /"MD"/ "train_data" / "3D_double_well_1e8steps_5e4burn_20261021_2056.npz"


#### chain of wells
DEFAULT_DATA = Path(__file__).resolve().parents[1] /"MD"/ "train_data" / "3D_chain_1e8steps_5e4burn_20261021_2133.npz"
#DEFAULT_DATA = Path(__file__).resolve().parents[1] /"MD"/ "train_data" / "2D_chain_1e8steps_5e4burn_20261019_2112.npz"


#"double_well_data_100000000steps_50000burn_20260918_115938.npz"

def main():
    #inputs 
    parser = argparse.ArgumentParser(description="Train a 2D position diffusion or flow model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=100_000)
    parser.add_argument("--generated-samples", type=int, default=1_000)
    parser.add_argument("--model", choices=("diffusion", "flow"), default="diffusion")
    parser.add_argument("--n-levels", type=int, default=12)
    parser.add_argument(
        "--steps-per-level", type=int, default=50,
        help="Score-based Langevin correction updates at each noise level (diffusion)",
    )
    parser.add_argument(
        "--ode-steps", type=int, default=50,
        help="Numerical integration steps over the flow model's transport interval",
    )
    parser.add_argument("--diffusion-frames", type=int, default=500_000)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.n_levels, args.steps_per_level, args.ode_steps) < 1:
        parser.error("--epochs, --n-levels, --steps-per-level, and --ode-steps must be positive")
    if args.generated_samples < 1:
        parser.error("--generated-samples must be at least 1")

    #data handling: load data train and compute mean/std for normalization
    train_data, true_samples, positions = load_equilibrium_bundle(
        args.data, args.max_samples, args.diffusion_frames)
    if train_data.ndim != 2 or train_data.shape[1] < 2:
        parser.error("The data archive must contain positions with x and y columns")
    train_data = train_data[:, :2]
    true_samples = true_samples[:, :2]
    positions = positions[:, :2]
    potential_type, barrier_amplitude, barrier_width = get_potential_settings(args.data)

    if len(train_data) < 2:
        parser.error("--max-samples must select at least 2 samples for training and validation")
    validation_size = max(1, int(len(train_data) * 0.1))
    permutation = torch.randperm(len(train_data), generator=torch.Generator().manual_seed(7))
    training_data = train_data[permutation[validation_size:]]
    validation_data = train_data[permutation[:validation_size]]
    mean = training_data.mean(dim=0).to(DEVICE)
    std = training_data.std(dim=0).clamp_min(1e-6).to(DEVICE)
    train_loader = DataLoader(TensorDataset(training_data), batch_size=512, shuffle=True)
    validation_loader = DataLoader(
        TensorDataset(validation_data), batch_size=512, shuffle=False,
    )
    
    #choose model type and train, generate samples
    if args.model == "flow":
        model, loss_history, validation_loss_history = train_flow(
            FlowModel().to(DEVICE), train_loader, mean, std,
            epochs=args.epochs,
            validation_loader=validation_loader,
            return_validation_history=True,
        )
        generated = sample_flow(model, mean, std, args.generated_samples, args.ode_steps)
        model_version = FLOW_VERSION
        sampling_config = f"ode_steps={args.ode_steps}"
        sampler_name = f"{args.ode_steps}ODESteps"
    else:
        model, loss_history, validation_loss_history = train_score(
            DiffusionModel().to(DEVICE), train_loader, mean, std,
            epochs=args.epochs,
            validation_loader=validation_loader,
            return_validation_history=True,
        )
        generated = sample_score(
            model, mean, std, args.generated_samples, args.n_levels, args.steps_per_level,
        )
        model_version = DIFFUSION_VERSION
        sampling_config = (
            f"noise_levels={args.n_levels};steps_per_level={args.steps_per_level}"
        )
        sampler_name = (
            f"{args.n_levels}NoiseLevels_{args.steps_per_level}StepsPerNoiseLevel"
        )
    

    #results: save plot, 'stats' and log file
    einstein = calculate_einstein_diffusion(positions)
    kramers_rate, barrier_height = calculate_kramers_rate(
        potential_type=potential_type,
        barrier_amplitude=barrier_amplitude,
        barrier_width=barrier_width,
    )
    marginal_distance, joint_distance = histogram_distances(true_samples, generated)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"{args.data.stem}_{args.model}_{args.epochs}epochs_"
        f"{len(train_data)}samples_{sampler_name}_{run_timestamp}"
    )
    prediction_dir = Path(__file__).resolve().parents[1] / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    output = prediction_dir / f"{run_name}_samples.png"
    loss_output = prediction_dir / f"{run_name}_loss.png"
    plot_results(
        true_samples,
        generated,
        output,
        model_type=args.model,
        potential_type=potential_type,
        barrier_amplitude=barrier_amplitude,
        barrier_width=barrier_width,
        show=args.show,
    )
    plot_training_loss(
        loss_history, loss_output, validation_losses=validation_loss_history, show=args.show,
    )
    log_path = log_prediction(
        output,
        model_version=model_version,
        einstein_diffusion=einstein,
        barrier_height=barrier_height,
        kramers_rate=kramers_rate,
        sampling_config=sampling_config,
        histogram_marginal_hellinger=marginal_distance,
        histogram_joint_hellinger=joint_distance,
    )
    print(f"Device: {DEVICE}")
    print(f"Einstein diffusion coefficient: {einstein:.5f}")
    print(f"Kramers barrier height: {barrier_height:.5f}")
    print(f"Kramers transition rate: {kramers_rate:.5f}")
    print(f"Mean marginal histogram Hellinger distance: {marginal_distance:.5f}")
    print(f"Joint histogram Hellinger distance: {joint_distance:.5f}")
    print(f"Visualization saved as '{output.resolve()}'")
    print(f"Training loss curve saved as '{loss_output.resolve()}'")
    print(f"Prediction log saved as '{log_path.resolve()}'")


if __name__ == "__main__":
    main()
