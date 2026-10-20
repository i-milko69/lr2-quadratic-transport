from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class OracleInfo:
    L: float | None = None
    mu: float | None = None
    x_star: np.ndarray | None = None
    f_star: float | None = None


class Oracle(ABC):
    """Базовый класс задачи оптимизации.

    Возвращает f(x), grad(x), и опционально L, mu, x_star, f_star.
    Если задача композитная — переопределите prox.
    """

    name: str = "oracle"
    dim: int = 0

    @abstractmethod
    def f(self, x: np.ndarray) -> float: ...

    @abstractmethod
    def grad(self, x: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def hess(self, x: np.ndarray) -> np.ndarray: ...

    @property
    def info(self) -> OracleInfo:
        return OracleInfo()

    def prox(self, x: np.ndarray, step: float) -> np.ndarray:
        """Proximal operator. По умолчанию — тождество."""
        return x

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name} dim={self.dim}>"

    def hess_diag(self, x: np.ndarray) -> np.ndarray:
        return None # np.diag(self.hess(x))

    def hess_vec(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
       return None # self.hess(x) @ v
