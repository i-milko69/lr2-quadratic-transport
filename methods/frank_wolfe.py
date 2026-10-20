"""Линеаризация для транспортной задачи.

Содержит:
  * FrankWolfe  — метод условного градиента (последовательная линеаризация).
  * lp_plan     — «LP-предел»: план классической (линейной) транспортной
                  задачи, т.е. b -> 0. Базовая линия для сравнения.

Идея Франка–Вульфа. На допустимом множестве
    X = {x >= 0 : sum_j x_ij = s_i, sum_i x_ij = d_j}
на каждой итерации линеаризуем f в текущей точке x_k и решаем ЛИНЕЙНУЮ
транспортную задачу
    v_k = argmin_{v in X} <grad f(x_k), v>,
после чего двигаемся к вершине: x_{k+1} = x_k + gamma_k (v_k - x_k).
Выпуклая комбинация допустимых точек допустима, поэтому штрафы за
ограничения и за x >= 0 НЕ нужны, а x_k допустим на каждой итерации.

ВАЖНО: оракул должен быть создан с mu = 0 (штрафов нет), иначе f и grad
содержат штрафные слагаемые, и метод будет решать не ту задачу.
Оракул обязан иметь атрибуты m, n, s, d (как у TransportOracle).

Критерий остановки — FW-зазор  gap_k = <grad f(x_k), x_k - v_k>.
Для выпуклой f верно  f(x_k) - f* <= gap_k, то есть зазор — это
гарантированная оценка субоптимальности без знания f*.
В History.grad_norm записывается именно зазор (обычная норма градиента
в условной задаче в минимуме не равна нулю).
"""
import numpy as np
from scipy.optimize import linprog

from .base import Method, History


def _constraint_matrix(m, n):
    """A_eq @ x = [s; d] для x, развёрнутого как x.reshape(m, n).ravel()."""
    rows = np.kron(np.eye(m), np.ones((1, n)))   # суммы по строкам (склады)
    cols = np.kron(np.ones((1, m)), np.eye(n))   # суммы по столбцам (магазины)
    return np.vstack([rows, cols])


def solve_transport_lp(cost, s, d):
    """min <cost, x>  s.t.  sum_j x_ij = s_i,  sum_i x_ij = d_j,  x >= 0."""
    cost = np.asarray(cost, dtype=float).ravel()
    m, n = len(s), len(d)
    res = linprog(
        cost,
        A_eq=_constraint_matrix(m, n),
        b_eq=np.concatenate([s, d]),
        bounds=(0, None),
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"LP не решён: {res.message}")
    return res.x


def lp_plan(oracle):
    """План классической транспортной задачи (b -> 0): min <a, x>."""
    return solve_transport_lp(oracle.a, oracle.s, oracle.d)


class FrankWolfe(Method):
    """Метод Франка–Вульфа (условный градиент).

    step_rule:
      'exact'   — точный линейный поиск (для квадратичной f — формула в
                  замкнутом виде), по умолчанию;
      'classic' — gamma_k = 2 / (k + 2).
    """

    def __init__(self, oracle, step_rule="exact", max_iter=1000, tol=1e-6,
                 step=None):
        super().__init__(oracle, max_iter=max_iter, tol=tol)
        if step_rule not in ("exact", "classic"):
            raise ValueError("step_rule: 'exact' | 'classic'")
        self.step_rule = step_rule
        self.name = f"frank_wolfe({step_rule})"
        self._A_eq = _constraint_matrix(oracle.m, oracle.n)
        self._b_eq = np.concatenate([oracle.s, oracle.d])

    # --- вспомогательное ---------------------------------------------------
    def _lmo(self, g):
        """Линейный минимизационный оракул: argmin_{v in X} <g, v>."""
        res = linprog(g, A_eq=self._A_eq, b_eq=self._b_eq,
                      bounds=(0, None), method="highs")
        if not res.success:
            raise RuntimeError(f"LMO не решён: {res.message}")
        return res.x

    def _is_feasible(self, x, tol=1e-8):
        return (np.all(x >= -tol)
                and np.allclose(self._A_eq @ x, self._b_eq, atol=tol))

    def _feasible_start(self, x0):
        """FW требует допустимой стартовой точки. x0 = 0 недопустим, поэтому
        берём x_ij = s_i d_j / sum(s) (допустимая внутренняя точка)."""
        x0 = np.asarray(x0, dtype=float).ravel()
        if self._is_feasible(x0):
            return x0
        o = self.oracle
        return (np.outer(o.s, o.d) / o.s.sum()).ravel()

    def _gamma(self, k, x, dvec, g):
        if self.step_rule == "classic":
            return 2.0 / (k + 2.0)
        H = self.oracle.hess(x)
        curv = float(dvec @ H @ dvec)
        if curv <= 0.0:
            return 1.0
        return float(np.clip(-(g @ dvec) / curv, 0.0, 1.0))

    # --- основной цикл -----------------------------------------------------
    def _step(self, state):  # требуется абстрактным классом; не используется
        raise NotImplementedError("FrankWolfe использует собственный run()")

    def run(self, x0):
        o = self.oracle
        x = self._feasible_start(x0)
        hist = History()

        g = o.grad(x)
        v = self._lmo(g)
        gap = float(g @ (x - v))
        hist.append(x, o.f(x), gap, 0.0)

        for k in range(1, self.max_iter + 1):
            if gap <= self.tol:
                break
            dvec = v - x
            gamma = self._gamma(k - 1, x, dvec, g)
            x = x + gamma * dvec
            g = o.grad(x)
            v = self._lmo(g)
            gap = float(g @ (x - v))
            hist.append(x, o.f(x), gap, gamma)
        return hist
