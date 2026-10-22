# Model package

 Run from `init_problem`:

```bash
../special_energy/bin/python -m model.run --epochs 1000 --max-samples 50000
```

The top-level script is a launcher. Everything used by the model is in this folder:

- `data.py`: loads positions, velocities, and energies.
- `diffusion_model.py`: small noise-conditioned score network.
- `flow_model.py`: small time-conditioned velocity network.
- `train.py`: trains the score network.
- `predict.py`: generates samples, calculates diagnostics, and makes plots.
- `run.py`: connects the files and runs the experiment.
- `future/model.py`: saved future experiment for time-step dynamics; it is not used yet.

Both models learn the 2D equilibrium distribution of positions `(x, y)` only. Neither receives physical MD time, velocity, or energy. Diffusion also receives its noise level and predicts a score; flow receives its internal transport coordinate and predicts a velocity. These internal coordinates are part of the generative algorithms, not physical trajectory time. Both use the same plots and diagnostics, so no separate prediction script is needed. Generated samples are saved in `init_problem/predictions/`.

Prediction figures overlay MD and generated position densities and potential-energy distributions. The generated-data legend identifies the selected model as either diffusion or flow.

The plotted potential is read from the dataset metadata. Older archives without this metadata are identified from their filename (`chain` or `double_well`); unrecognized names raise an error instead of silently plotting the wrong potential. H-barrier plots use the saved ridge amplitude and width.

Each new prediction-log entry also records mean marginal and joint histogram Hellinger distances between reference and generated samples. Both range from 0 (matching histograms) to 1 (maximally different); the CSV begins with brief metric definitions. These are logged metrics only and do not alter the existing 2D plots. Sampler settings are included in the log and output filename: diffusion records noise levels and steps per level, while flow records ODE integration steps.

Both `--steps-per-level` and `--ode-steps` default to 50. Diffusion applies that many stochastic score-based correction updates at each noise level (600 updates total with the default 12 noise levels); flow uses that many numerical integration steps to follow its learned velocity field from noise to samples. Neither is a physical MD-time step.

## 3D samples and trajectory transitions

The separate 3D runner requires an archive with `positions` and `md_time` arrays:

```bash
python -m model.run_3D --data MD/train_data/3D_double_well_data.npz --objective joint
python -m model.run_3D --data MD/train_data/3D_double_well_data.npz --objective transitions --model flow
```

`--objective joint` trains diffusion or flow on shuffled `(x, y, physical MD time)` samples. It learns their joint distribution, not temporal order. `--objective transitions` instead trains a conditional distribution of the next `(x, y)` from the current `(x, y, t)` using consecutive saved frames, then rolls it forward to estimate basin-crossing rates. The transition objective uses the saved chronology; the joint objective does not. Its crossing-rate estimate counts basin-label changes per unit physical time (two basins split at `x=0` for double wells; 16 periodic bins for the chain), so it is a sampled discrete-time diagnostic, not a guaranteed kinetic rate. Generated arrays, a task-specific comparison figure, and training loss are saved separately. The existing 2D equilibrium pipeline remains unchanged.

Generate a high-barrier double-well trajectory with:

```bash
python MD/MD_sim.py --potential double-well-Hbarrier
```

This adds a Gaussian ridge along `x=0`, with default width `0.25`. Its amplitude is solved using harmonic transition-state theory so the expected number of crossings in the first half of the saved trajectory is one. The target rate is `k = 1 / (T_saved / 2)`. The calculation uses the two-dimensional harmonic TST prefactor (including the stable transverse frequency at the saddle); for the default `100,000,000` steps, `dt=0.005`, `50,000` burn-in steps, and `kBT=0.4`, it gives a barrier of about `4.99` energy units above the well and a ridge amplitude about `2.90`. These are TST estimates, not a guarantee of exactly one observed transition: friction and recrossings can change the realized count. Override the defaults with `--target-transitions-half` and `--barrier-width`.

Training reserves a deterministic 10% of the selected samples for validation. The score-matching objective weights errors by the squared noise level, and fixed validation noise makes the validation curve comparable across epochs. The loss plot contains both training and validation loss by epoch.

For a quick local run:

```bash
../special_energy/bin/python -m model.run --epochs 500 --max-samples 10000
```

Select flow matching with `--model flow`; the default is `--model diffusion`.

Both networks are deliberately small MLPs. The diffusion implementation is a compact score-based baseline, not a full DDPM/SDE solver. Einstein diffusion and the Kramers rate remain MD-trajectory diagnostics; they are not predicted by either 2D model.

Each PNG in `init_problem/predictions/` is registered in `prediction_log.csv` with its model version, Einstein diffusion, Kramers barrier height, Kramers transition rate, and device.
