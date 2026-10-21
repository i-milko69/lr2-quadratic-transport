import numpy as np
from .gd import GD


class PreconditionedGD(GD):
    """GD с диагональным предобуславливанием (Якоби).

        x_{k+1} = x_k - eta * g / diag(H(x_k))

    step=None: безопасный шаг выбирается автоматически (см. _default_step).
    Устойчивость требует eta * lambda_max(D^-1/2 H D^-1/2) < 2, а для штрафной
    транспортной задачи lambda_max ~ (m+n)/2, то есть фиксированное eta=0.1
    расходится уже при m+n > 40.
    """

    def __init__(self, oracle, step=None, max_iter=1000, tol=1e-8, eps=1e-12):
        if step is None:
            step = self._default_step(oracle)
        super().__init__(oracle, step=step, max_iter=max_iter, tol=tol)
        self.eps = eps
        self.name = "preconditioned_gd"

    @staticmethod
    def _default_step(oracle):
        m, n = getattr(oracle, "m", None), getattr(oracle, "n", None)
        if m is not None and n is not None:
            # lambda_max(D^-1/2 H D^-1/2) <= (m+n)/2 + 1 для штрафного F_mu
            return 1.0 / ((m + n) / 2.0 + 1.0)
        # общий случай: оценка по Гессиану в нуле
        H = oracle.hess(np.zeros(oracle.dim))
        if H is None:
            raise ValueError("Для метода нужна матрица Гессе.")
        d = np.maximum(np.diag(H), 1e-12)
        return 1.0 / float(np.linalg.eigvalsh(H / np.sqrt(np.outer(d, d)))[-1])

    def _diag(self, x):
        d = self.oracle.hess_diag(x)          # O(dim), если оракул умеет
        if d is None:
            H = self.oracle.hess(x)           # запасной путь: полная матрица
            if H is None:
                raise ValueError("Для метода нужна матрица Гессе.")
            d = np.diag(H)
        return np.maximum(d, self.eps)

    def _step(self, state):
        x = state["x"]
        g = self.oracle.grad(x)
        diag = self._diag(x)
        eta = self._resolve_step(x, g, state)

        state["x_prev"] = x.copy()
        state["x"] = x - eta * g / diag
        state["step"] = eta

        return state
