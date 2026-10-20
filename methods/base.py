from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np


@dataclass
class History:
    x: list = field(default_factory=list)
    f: list = field(default_factory=list)
    grad_norm: list = field(default_factory=list)
    step_size: list = field(default_factory=list)

    def append(self, x, f, g, step):
        self.x.append(np.copy(x))
        self.f.append(float(f))
        self.grad_norm.append(float(g))
        self.step_size.append(float(step))


class Method(ABC):
    """Базовый итеративный метод."""

    def __init__(self, oracle, step=None, max_iter: int = 1000,
                 tol: float = 1e-8):
        self.oracle = oracle
        self.max_iter = max_iter
        self.tol = tol
        self.name = self.__class__.__name__.lower()

    @abstractmethod
    def _step(self, state: dict) -> dict: ...

    def run(self, x0: np.ndarray) -> History:
        x0 = np.asarray(x0, dtype=float)
        state = {"x": x0.copy(), "x_prev": x0.copy(), "k": 0, "step": 0.0}
        hist = History()

        f0 = self.oracle.f(state["x"])
        g0 = np.linalg.norm(self.oracle.grad(state["x"]))
        hist.append(state["x"], f0, g0, 0.0)

        for k in range(1, self.max_iter + 1):
            state = self._step(state)
            state["k"] = k
            f = self.oracle.f(state["x"])
            g = np.linalg.norm(self.oracle.grad(state["x"]))
            hist.append(state["x"], f, g, state.get("step", 0.0))
            if g <= self.tol:
                break
        return hist