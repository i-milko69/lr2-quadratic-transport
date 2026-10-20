# Configs

Each YAML file is one experiment. Run it with:

    python -m scripts.run_experiment --config configs/quadratic_gd.yaml

Sweeps are in `sweeps/` and run with:

    python -m scripts.run_sweep --sweep configs/sweeps/kappa.yaml

## Fields

- `oracle.type` — quadratic | logistic
- `oracle.*` — task parameters (dim, kappa, n, l2, seed)
- `method.type` — gd | adaptive
- `method.*` — method parameters (step, strategy, eta0, max_iter, tol)
- `run.x0` — zeros | random
- `seed` — global seed
- `out_dir` — where JSON logs are written