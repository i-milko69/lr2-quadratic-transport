"""Эталонное решение исходной (без штрафов) транспортной задачи."""
import numpy as np
from scipy.optimize import minimize

from methods.frank_wolfe import _constraint_matrix


def reference_solution(oracle):
    """Решает min sum(a x + b x^2/2) при всех ограничениях задачи через SLSQP.

    Режим ёмкостей: sum_j x_ij <= C_i, sum_i x_ij = d_j, x >= 0.
    Классический режим: оба набора равенств.
    Возвращает (x, f*).
    """
    m, n = oracle.m, oracle.n
    M = _constraint_matrix(m, n)
    a, b = oracle.a.ravel(), oracle.b.ravel()
    cap = getattr(oracle, "capacity", None)

    if cap is None:
        rhs = np.concatenate([oracle.s, oracle.d])
        cons = [{"type": "eq", "fun": lambda x: M @ x - rhs, "jac": lambda x: M}]
    else:
        cons = [
            {"type": "ineq", "fun": lambda x: cap - M[:m] @ x, "jac": lambda x: -M[:m]},
            {"type": "eq", "fun": lambda x: M[m:] @ x - oracle.d, "jac": lambda x: M[m:]},
        ]
    x0 = np.outer(oracle.s, oracle.d).ravel() / oracle.s.sum()
    res = minimize(lambda x: a @ x + 0.5 * np.sum(b * x**2), x0,
                   jac=lambda x: a + b * x, bounds=[(0, None)] * (m * n),
                   constraints=cons, method="SLSQP",
                   options={"ftol": 1e-12, "maxiter": 1000})
    x = res.x
    # SLSQP может сообщить о потере точности у границы; важна допустимость.
    xm = x.reshape(m, n)
    viol = np.abs(xm.sum(axis=0) - oracle.d).max()
    viol = max(viol, (xm.sum(axis=1) - cap).max() if cap is not None
               else np.abs(xm.sum(axis=1) - oracle.s).max(), -x.min())
    if viol > 1e-6:
        raise RuntimeError(f"SLSQP: решение недопустимо ({viol:.1e}): {res.message}")
    return x, float(res.fun)
