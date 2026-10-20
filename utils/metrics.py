import time
import numpy as np


class Timer:
    """Context manager to measure wall-clock time."""

    def __init__(self):
        self.elapsed = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._t0


def grad_norm(oracle, x: np.ndarray) -> float:
    return float(np.linalg.norm(oracle.grad(x)))


def suboptimality(oracle, x: np.ndarray) -> float | None:
    """f(x) - f*, if f* is known."""
    f_star = oracle.info.f_star
    if f_star is None:
        return None
    return float(oracle.f(x) - f_star)


def distance_to_opt(oracle, x: np.ndarray) -> float | None:
    x_star = oracle.info.x_star
    if x_star is None:
        return None
    return float(np.linalg.norm(x - x_star))


def condition_number(oracle, x: np.ndarray) -> float | None:
    """Local condition number from Hessian, if available."""
    if not oracle.has_hess:
        return None
    H = oracle.hess(x)
    eigvals = np.linalg.eigvalsh(H)
    if eigvals.min() <= 0:
        return float("inf")
    return float(eigvals.max() / eigvals.min())


def iterations_to_tol(history, tol: float = 1e-6) -> int | None:
    """First iteration where grad_norm <= tol."""
    for k, g in enumerate(history.grad_norm):
        if g <= tol:
            return k
    return None


def summarize(history, oracle=None, tol: float = 1e-6) -> dict:
    """Standard summary dict for logs and tables."""
    out = {
        "n_iter": len(history.f),
        "f_final": history.f[-1] if history.f else None,
        "grad_norm_final": history.grad_norm[-1] if history.grad_norm else None,
        "iters_to_tol": iterations_to_tol(history, tol),
    }
    if oracle is not None and history.x:
        x_last = history.x[-1]
        out["suboptimality"] = suboptimality(oracle, x_last)
        out["dist_to_opt"] = distance_to_opt(oracle, x_last)
    return out