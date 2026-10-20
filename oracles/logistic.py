
import numpy as np
from .base import Oracle, OracleInfo


class LogisticOracle(Oracle):
    """L2-регуляризованная логистическая регрессия.

    Функция потерь:
    f(w) = среднее log(1 + exp(-y_i * X_i @ w))
           + l2 / 2 * ||w||^2
    """

    def __init__(self, n: int = 1000, dim: int = 50,
                 l2: float = 1.0, seed: int = 0):
        # Генерируем случайные данные
        rng = np.random.default_rng(seed)
        self.X = rng.standard_normal((n, dim))
        self.y = np.sign(rng.standard_normal(n))
        self.y[self.y == 0] = 1.0

        self.l2 = l2
        self.dim = dim
        self.name = f"logistic(n={n}, dim={dim}, l2={l2:g})"

        # Оцениваем константу Липшица градиента
        sigma_max = np.linalg.norm(self.X, ord=2)
        self._L = sigma_max ** 2 / (4.0 * n) + l2
        self._mu = l2

    def f(self, w: np.ndarray) -> float:
        # Вычисляем логистическую функцию потерь
        z = -self.y * (self.X @ w)
        loss = np.mean(np.logaddexp(0.0, z))
        return float(loss + 0.5 * self.l2 * (w @ w))

    def grad(self, w: np.ndarray) -> np.ndarray:
        # Вычисляем градиент функции потерь
        z = -self.y * (self.X @ w)

        # Устойчивая формула для производной логистической функции
        sig = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

        # Производная средней функции потерь
        g = -(self.X.T @ (self.y * sig)) / len(self.y)
        return g + self.l2 * w

    def hess(self, w: np.ndarray) -> np.ndarray:
        # Матрица Гессе логистической функции потерь
        z = -self.y * (self.X @ w)
        sig = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

        # Веса для матрицы Гессе
        weights = sig * (1.0 - sig)

        # Добавляем L2-регуляризацию
        H = (self.X.T * weights) @ self.X / len(self.y)
        H += self.l2 * np.eye(self.dim)
        return H

    @property
    def info(self):
        return OracleInfo(L=self._L, mu=self._mu)
