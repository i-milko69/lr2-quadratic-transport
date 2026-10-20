import numpy as np
from .base import Oracle, OracleInfo


class QuadraticOracle(Oracle):
    """f(x) = 1/2 x^T A x - b^T x.

    A — симметричная положительно определённая с заданным числом
    обусловленности kappa = L / mu.
    """

    def __init__(self, dim: int = 100, kappa: float = 100.0,
                 L: float = 1.0, seed: int = 0):
        rng = np.random.default_rng(seed)
        Q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
        mu = L / kappa
        eigs = np.linspace(mu, L, dim) # TODO: add random distribution of task curvate
        A = Q @ np.diag(eigs) @ Q.T
        self.A = 0.5 * (A + A.T)
        self.b = rng.standard_normal(dim)
        self.dim = dim
        self.name = f"quadratic(dim={dim}, kappa={kappa:g})"
        self._L = L
        self._mu = mu
        self._x_star = np.linalg.solve(self.A, self.b)
        self._f_star = 0.5 * self._x_star @ self.A @ self._x_star - self.b @ self._x_star

    def f(self, x: np.ndarray) -> float:
        return 0.5 * x @ self.A @ x - self.b @ x

    def grad(self, x: np.ndarray) -> np.ndarray:
        return self.A @ x - self.b

    def hess(self, x: np.ndarray) -> np.ndarray:
        return self.A

    @property
    def info(self) -> OracleInfo:
        return OracleInfo(L=self._L, mu=self._mu,
                          x_star=self._x_star, f_star=self._f_star)
