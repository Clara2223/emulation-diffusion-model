# Model package

 Run from `init_problem`:

```bash
../special_energy/bin/python -m model.run --train-steps 1000 --max-samples 50000
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

For a quick local run:

```bash
../special_energy/bin/python -m model.run --train-steps 500 --max-samples 10000
```

Select flow matching with `--model flow`; the default is `--model diffusion`.

Both networks are deliberately small MLPs. The diffusion implementation is a compact score-based baseline, not a full DDPM/SDE solver. Einstein diffusion and the Kramers rate remain MD-trajectory diagnostics; they are not predicted by either 2D model.

Each PNG in `init_problem/predictions/` is registered in `prediction_log.csv` with its model version, Einstein diffusion, Kramers barrier height, Kramers transition rate, and device.
