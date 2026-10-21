import numpy as np
from .base import Method, History


class ProxGD(Method):
    """Proximal gradient descent with projection onto x >= 0."""

    def __init__(self, oracle, step=None, max_iter=10000, tol=1e-7):
        super().__init__(oracle, step=step, max_iter=max_iter, tol=tol)

        if step is None:
            L = oracle.info.L
            step = 1.0 / L if L and L > 0 else 1e-3

        self.step = float(step)
        self.name = "prox_gd"

    def _step(self, state):
        x = state["x"]
        grad = self.oracle.grad(x)

        # Gradient step, then projection onto the nonnegative orthant.
        x_new = np.maximum(0.0, x - self.step * grad)

        state["x_prev"] = x.copy()
        state["x"] = x_new
        state["step"] = self.step
        return state

    def run(self, x0):
        x = np.asarray(x0, dtype=float).copy()
        history = History()

        for k in range(self.max_iter + 1):
            grad = self.oracle.grad(x)

            # Projected-gradient mapping: stationarity measure
            # for optimization with the constraint x >= 0.
            x_projected = np.maximum(0.0, x - self.step * grad)
            stationarity = np.linalg.norm((x - x_projected) / self.step)

            history.append(
                x,
                self.oracle.f(x),
                stationarity,
                0.0 if k == 0 else self.step,
            )

            if stationarity <= self.tol or k == self.max_iter:
                break

            x = x_projected

        return history
