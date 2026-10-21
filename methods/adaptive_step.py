import numpy as np
from .base import Method


class AdaptiveStep(Method):
    """Обёртка над любым методом с адаптивным шагом.

    Стратегии:
      'polyak'     — eta = (f(x) - f*) / ||grad||^2
      'bb1'        — Barzilai-Borwein 1
      'bb2'        — Barzilai-Borwein 2
      'backtrack'  — Armijo backtracking
    """

    def __init__(self, oracle, strategy="backtrack", eta0=1.0,
                 alpha=0.5, beta_arm=1e-4, max_iter=1000, tol=1e-8):
        super().__init__(oracle, max_iter=max_iter, tol=tol)
        self.strategy = strategy
        self.eta0 = eta0
        self.alpha = alpha
        self.beta_arm = beta_arm
        self.name = f"adaptive_step({strategy})"
        self._eta = eta0
        self._prev_grad = None
        self._prev_x = None

    def _polyak(self, x, g, f) -> float:
        f_star = self.oracle.info.f_star
        if f_star is None:
            return self._eta
        denom = float(g @ g) + 1e-12
        return max((f - f_star) / denom, 1e-12)

    def _bb(self, x, g, mode: int) -> float:
        if self._prev_grad is None:
            return self._eta
        s = x - self._prev_x
        y = g - self._prev_grad
        if mode == 1:
            denom = float(s @ y) + 1e-12
            return float(s @ s) / denom
        denom = float(y @ y) + 1e-12
        return float(s @ y) / denom

    def _backtrack(self, x, g, direction) -> float:
        eta = self._eta
        fx = self.oracle.f(x)
        gx = float(g @ direction)
        for _ in range(50):
            x_new = x + eta * direction
            if self.oracle.f(x_new) <= fx + self.beta_arm * eta * gx:
                break
            eta *= self.alpha
        return eta

    def _step(self, state: dict) -> dict:
        x = state["x"]
        g = self.oracle.grad(x)
        f = self.oracle.f(x)
        direction = -g

        if self.strategy == "polyak":
            step = self._polyak(x, g, f)
        elif self.strategy == "bb1":
            step = self._bb(x, g, mode=1)
        elif self.strategy == "bb2":
            step = self._bb(x, g, mode=2)
        else:
            step = self._backtrack(x, g, direction)

        self._prev_x = x.copy()
        self._prev_grad = g.copy()
        self._eta = step

        state["x_prev"] = x
        state["x"] = x + step * direction
        state["step"] = step
        return state
