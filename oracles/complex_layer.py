import numpy as np
from .base import Oracle


class HuberOracle(Oracle):
    """Робастная регрессия с потерей Хьюбера.

    f(w) = 1/n sum rho_delta(r_i), r_i = x_i^T w - y_i
    rho_delta(r) = 0.5 r^2,               |r| <= delta
                 = delta(|r| - 0.5 delta), иначе

    Гладкая выпуклая задача с меняющейся кривизной.
    """

    def __init__(self, n: int = 1000, dim: int = 20,
                 delta: float = 1.0, outlier_frac: float = 0.05,
                 seed: int = 0):
        rng = np.random.default_rng(seed)
        self.X = rng.standard_normal((n, dim))
        w_true = rng.standard_normal(dim)
        self.y = self.X @ w_true + 0.1 * rng.standard_normal(n)

        n_out = int(n * outlier_frac)
        if n_out > 0:
            idx = rng.choice(n, n_out, replace=False)
            self.y[idx] += 5.0 * rng.standard_normal(n_out)

        self.delta = delta
        self.dim = dim
        self.name = f"huber(n={n}, dim={dim}, delta={delta:g})"

        sigma_max = np.linalg.norm(self.X, ord=2)
        self._L = sigma_max ** 2 / n

    def f(self, w: np.ndarray) -> float:
        r = self.X @ w - self.y
        a = np.abs(r)
        small = a <= self.delta
        loss = np.where(small,
                        0.5 * r ** 2,
                        self.delta * (a - 0.5 * self.delta))
        return float(np.mean(loss))

    def grad(self, w: np.ndarray) -> np.ndarray:
        r = self.X @ w - self.y
        small = np.abs(r) <= self.delta
        coef = np.where(small, r, self.delta * np.sign(r))
        return self.X.T @ coef / len(self.y)

    def hess(self, x: np.ndarray) -> np.ndarray:
        return None