
import numpy as np
from .gd import GD


class PreconditionedGD(GD):
    """Gradient descent with diagonal Hessian preconditioning."""

    def __init__(
        self,
        oracle,
        step=0.1,
        max_iter=1000,
        tol=1e-8,
        eps=1e-12,
    ):
        super().__init__(
            oracle,
            step=step,
            max_iter=max_iter,
            tol=tol,
        )
        self.eps = eps
        self.name = "preconditioned_gd"

    def _step(self, state):
        x = state["x"]
        g = self.oracle.grad(x)

        H = self.oracle.hess(x)
        if H is None:
            raise ValueError("Для метода нужна матрица Гессе.")

        diag = np.maximum(np.diag(H), self.eps)
        eta = self._resolve_step(x, g, state)

        state["x_prev"] = x.copy()
        state["x"] = x - eta * g / diag
        state["step"] = eta

        return state
