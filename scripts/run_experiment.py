"""Run one experiment from a YAML config and save JSON logs.

Embedded sanity checks (a minimal test system) run before the
main experiment unless --no-checks is passed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from oracles import QuadraticOracle, LogisticOracle, TransportOracle
from methods import GD, AdaptiveStep, FrankWolfe, PreconditionedGD

# ---------------------- checks ----------------------

def check_grad_finite_diff(oracle, x, eps=1e-6, tol=1e-4):
    """Analytic gradient must match central finite differences."""
    g_an = oracle.grad(x)
    g_fd = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x)
        e[i] = eps
        g_fd[i] = (oracle.f(x + e) - oracle.f(x - e)) / (2 * eps)
    err = float(np.max(np.abs(g_an - g_fd)))
    assert err < tol, f"grad mismatch: {err:.2e} > {tol:.2e}"
    return err


def check_hess_psd(oracle, x, tol=1e-8):
    """Hessian must be PSD for convex problems. Returns None if no Hessian."""
    H = oracle.hess(x)
    if H is None:
        return None
    eigs = np.linalg.eigvalsh(H)
    lam_min = float(eigs.min())
    assert lam_min >= -tol, f"hess not PSD: min eig = {lam_min:.2e}"
    return lam_min


def check_gd_decreases_loss(oracle, x0, step, n_iter=200, tol_noise=1e-8):
    """Loss must not increase along GD iterates."""
    h = GD(oracle, step=step, max_iter=n_iter, tol=0.0).run(x0)
    diffs = np.diff(h.f)
    worst = float(diffs.max()) if len(diffs) else 0.0
    assert worst <= tol_noise, f"loss increased: max diff = {worst:.2e}"
    return h


def check_gd_converges(oracle, x0, step, tol=1e-4, n_iter=5000):
    """Gradient norm must drop below tol within n_iter."""
    h = GD(oracle, step=step, max_iter=n_iter, tol=tol).run(x0)
    g_final = float(h.grad_norm[-1])
    assert g_final <= tol, \
        f"GD did not converge: ||grad|| = {g_final:.2e} > {tol:.2e}"
    return h


def run_checks(cfg):
    """Lightweight sanity checks against the config's oracle.

    Fixed initial conditions: x0 = zeros, plus a deterministic random
    probe for gradient/Hessian checks.
    """
    oracle = _build_oracle(cfg["oracle"])
    x_probe = np.random.default_rng(0).standard_normal(oracle.dim)
    x0 = np.zeros(oracle.dim)

    print(f"[checks] oracle = {oracle.name}")

    err = check_grad_finite_diff(oracle, x_probe)
    print(f"[checks] grad vs finite-diff: max err = {err:.2e}")

    lam_min = check_hess_psd(oracle, x_probe)
    if lam_min is not None:
        print(f"[checks] hess PSD: min eig = {lam_min:.2e}")
    else:
        print("[checks] hess not available, skipped")

    L = oracle.info.L
    step = 1.0 / L if L else 1e-3
    check_gd_decreases_loss(oracle, x0, step=step, n_iter=200)
    print("[checks] GD decreases loss: OK")
    
    tol = float(cfg["method"].get("tol", 1e-4))
    n_iter = int(cfg["method"].get("max_iter", 5000))

    check_gd_converges(oracle, x0, step=step, tol=tol, n_iter=n_iter)
    print("[checks] GD converges: OK")


# ---------------------- runner ----------------------
ORACLES = {
    "quadratic": QuadraticOracle,
    "logistic": LogisticOracle,
    "transport": TransportOracle,
}
METHODS = {"gd": GD, "adaptive": AdaptiveStep, "fw": FrankWolfe, "pgd": PreconditionedGD}


def _build_oracle(cfg):
    return ORACLES[cfg["type"]](**{k: v for k, v in cfg.items() if k != "type"})


def _build_method(cfg, oracle):
    params = {k: v for k, v in cfg.items() if k != "type"}
    return METHODS[cfg["type"]](oracle, **params)


def run_experiment(cfg):
    seed = cfg.get("seed", 42)
    np.random.seed(seed)

    oracle = _build_oracle(cfg["oracle"])
    method = _build_method(cfg["method"], oracle)

    # fixed initial conditions
    x0 = np.zeros(oracle.dim)
    history = method.run(x0)

    return {"oracle": oracle, "method": method, "history": history, "config": cfg}


def save_results(result, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = result["config"]
    name = (f"{cfg['oracle']['type']}__{cfg['method']['type']}"
            f"__seed{cfg.get('seed', 42)}.json")
    path = out / name

    hist = result["history"]
    payload = {
        "config": cfg,
        "f": hist.f,
        "grad_norm": hist.grad_norm,
        "step_size": hist.step_size,
        "n_iter": len(hist.f),
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default="results/logs")
    parser.add_argument("--no-checks", action="store_true",
                        help="Skip embedded sanity checks.")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    if not args.no_checks:
        run_checks(cfg)

    result = run_experiment(cfg)
    path = save_results(result, args.out)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
