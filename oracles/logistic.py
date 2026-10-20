import numpy as np
from .base import Oracle


class LogisticOracle(Oracle):
    """L2-регуляризованная логистическая регрессия.

    f(w) = 1/n sum log(1 + exp(-y_i * w^T x_i)) + l2/2 * ||w||^2
    """

    def __init__(self, n: int = 1000, dim: int = 50,
                 l2: float = 1.0, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.X = rng.standard_normal((n, dim))
        self.y = np.sign(rng.standard_normal(n))
        self.y[self.y == 0] = 1.0
        self.l2 = l2
        self.dim = dim
        self.name = f"logistic(n={n}, dim={dim}, l2={l2:g})"

        # грубая оценка гладкости: ||X||^2 / (4n) + l2
        sigma_max = np.linalg.norm(self.X, ord=2)
        self._L = sigma_max ** 2 / (4.0 * n) + l2
        self._mu = l2  # нижняя граница

    def f(self, w: np.ndarray) -> float:
        z = -self.y * (self.X @ w)
        loss = # TODO
        return loss + 0.5 * self.l2 * w @ w

    def grad(self, w: np.ndarray) -> np.ndarray:
        z = # TODO
        sig = 1.0 / (1.0 + np.exp(-z))       # sigmoid(z)
        g = # TODO
        return g + self.l2 * w
    
    def hess(self, x: np.ndarray) -> np.ndarray:
        return # TODO
