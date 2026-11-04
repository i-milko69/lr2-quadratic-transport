import numpy as np
from .base import Method, History


class ProxGD(Method):
    """Проксимальный градиентный спуск.

        x_{k+1} = prox_{step * g}(x_k - step * grad f(x_k)),

    где g — индикатор допустимого множества X (x >= 0 и, если заданы,
    ёмкости складов sum_j x_ij <= C_i), то есть prox = проекция на X.
    Проекция берётся у оракула: oracle.prox(x, step).

    Остановка по норме проксимального отображения градиента
        ||(x - prox(x - step*grad)) / step|| <= tol,
    она обращается в 0 ровно в точках, удовлетворяющих условиям оптимальности
    задачи с ограничением (обычная ||grad|| в них не равна 0).
    """

    def __init__(self, oracle, step=None, max_iter=10000, tol=1e-7):
        super().__init__(oracle, step=step, max_iter=max_iter, tol=tol)

        if step is None:
            L = oracle.info.L
            step = 1.0 / L if L and L > 0 else 1e-3

        self.step = float(step)
        self.name = "prox_gd"

    def _prox_step(self, x):
        return self.oracle.prox(x - self.step * self.oracle.grad(x), self.step)

    def _step(self, state):
        x = state["x"]
        state["x_prev"] = x.copy()
        state["x"] = self._prox_step(x)
        state["step"] = self.step
        return state

    def run(self, x0):
        # Стартовую точку сразу проецируем: все итерации допустимы по x и C.
        x = self.oracle.prox(np.asarray(x0, dtype=float).copy(), self.step)
        history = History()

        for k in range(self.max_iter + 1):
            x_next = self._prox_step(x)
            stationarity = np.linalg.norm((x - x_next) / self.step)

            history.append(x, self.oracle.f(x), stationarity,
                           0.0 if k == 0 else self.step)

            if stationarity <= self.tol or k == self.max_iter:
                break
            x = x_next

        return history
