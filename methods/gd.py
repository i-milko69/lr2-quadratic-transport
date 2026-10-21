import numpy as np
from .base import Method


class GD(Method):
    """Gradient Descent:
    x_{k+1} = x_k - eta * grad f(x_k).

    step: float or callable(oracle, x, g, state) -> float
    """

    def __init__(self, oracle, step=None, max_iter=1000, tol=1e-8):
        super().__init__(oracle, max_iter=max_iter, tol=tol)

        if step is None:
            L = oracle.info.L
            step = 1.0 / L if L else 1e-3

        self.step = step
        self.name = "gd"

    def _resolve_step(self, x, g, state):
        if np.isscalar(self.step):
            return float(self.step)

        return float(self.step(self.oracle, x, g, state))

    def _step(self, state):
        x = state["x"]
        g = self.oracle.grad(x)
        eta = self._resolve_step(x, g, state)

        state["x_prev"] = x.copy()
        state["x"] = x - eta * g
        state["step"] = eta

        return state
