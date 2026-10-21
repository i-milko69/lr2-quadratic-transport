"""Continuation по mu против обычного GD при РАВНОЙ точности.

Запуск из корня репозитория:  python -m scripts.compare_continuation

Критерий остановки — разность значений штрафной функции с её точным минимумом,
F_mu(x) - F_mu* <= eps (считается ньютоновским методом). Она не зависит от
масштабирования переменных, поэтому число итераций сравнимо между вариантами.
Главный параметр continuation — точность решения промежуточных этапов, его и
перебираем.
"""
import numpy as np
from scipy.optimize import minimize

from oracles.transport import TransportOracle
from methods.gd import GD

M, N, SEED = 3, 3, 42
MU_LADDER = (10, 100, 1000)       # последний — целевой штраф
EPS = 1e-4                         # точность на целевом mu
EPS_MID = (1e-1, 1e-2, 1e-4)       # точность промежуточных этапов
MAX_ITER = 400_000


def newton_min(oracle, x0, tol=1e-11, max_iter=200):
    """Точный минимум F_mu (демпфированный Ньютон) — эталон для F_mu*."""
    x = np.array(x0, dtype=float)
    for _ in range(max_iter):
        g = oracle.grad(x)
        if np.linalg.norm(g) < tol:
            break
        p = -np.linalg.solve(oracle.hess(x), g)
        t, f0 = 1.0, oracle.f(x)
        while oracle.f(x + t * p) > f0 + 1e-4 * t * (g @ p) and t > 1e-14:
            t *= 0.5
        x = x + t * p
    return x


def iters_to_gap(oracle, x0, F_star, eps, max_iter=MAX_ITER, chunk=2000):
    """GD (шаг 1/L) до F(x) - F_star <= eps. Возвращает (x, итерации, ok)."""
    x, total = np.array(x0, dtype=float), 0
    while total < max_iter:
        n = min(chunk, max_iter - total)
        h = GD(oracle, max_iter=n, tol=0.0).run(x)
        hit = np.flatnonzero(np.array(h.f) - F_star <= eps)
        if hit.size:
            return h.x[hit[0]], total + int(hit[0]), True
        x, total = h.x[-1], total + n
    return x, total, False


def constrained_reference(oracle):
    """f*, x* исходной задачи с ограничениями (SLSQP)."""
    m, n = oracle.m, oracle.n
    A = np.vstack([np.kron(np.eye(m), np.ones((1, n))),
                   np.kron(np.ones((1, m)), np.eye(n))])[:-1]   # баланс => одно лишнее
    r = np.concatenate([oracle.s, oracle.d])[:-1]
    a, b = oracle.a.ravel(), oracle.b.ravel()
    res = minimize(lambda x: a @ x + 0.5 * (b * x**2).sum(), np.full(m * n, 100.0 / (m * n)),
                   jac=lambda x: a + b * x, method="SLSQP", bounds=[(0, None)] * (m * n),
                   constraints=[{"type": "eq", "fun": lambda x: A @ x - r, "jac": lambda x: A}],
                   options={"maxiter": 500, "ftol": 1e-14})
    return res.x, float(a @ res.x + 0.5 * (b * res.x**2).sum())


def constraint_residual(oracle, x):
    X = np.asarray(x).reshape(oracle.m, oracle.n)
    return max(np.max(np.abs(X.sum(axis=1) - oracle.s)),
               np.max(np.abs(X.sum(axis=0) - oracle.d)))


def continuation(eps_mid):
    x, total = np.zeros(M * N), 0
    for mu in MU_LADDER:
        o = TransportOracle(m=M, n=N, seed=SEED, mu=mu)   # L пересчитывается в __init__
        F_star = o.f(newton_min(o, x))
        x, it, _ = iters_to_gap(o, x, F_star, EPS if mu == MU_LADDER[-1] else eps_mid)
        total += it
    return x, total


def main():
    mu_t = MU_LADDER[-1]
    final = TransportOracle(m=M, n=N, seed=SEED, mu=mu_t)
    cost = TransportOracle(m=M, n=N, seed=SEED, mu=0.0)     # исходная стоимость без штрафов
    x_star, f_star = constrained_reference(final)
    x0 = np.zeros(final.dim)

    runs = []
    x, it, ok = iters_to_gap(final, x0, final.f(newton_min(final, x0)), EPS)
    runs.append(("обычный GD", x, it, ok))
    for e in EPS_MID:
        x, it = continuation(e)
        runs.append((f"continuation, этапы до {e:g}", x, it, True))

    print(f"M={M} N={N} seed={SEED}; целевой mu={mu_t}; точность F_mu - F_mu* <= {EPS:g}")
    print(f"f* исходной задачи (с ограничениями) = {f_star:.6f}\n")
    print(f"{'вариант':<30}{'итераций':>10}{'f - f*':>12}{'невязка':>11}{'|x-x*|':>10}{'|grad|':>10}")
    for name, x, it, ok in runs:
        print(f"{name:<30}{it:>10d}{cost.f(x) - f_star:>12.4g}{constraint_residual(cost, x):>11.2e}"
              f"{np.linalg.norm(x - x_star):>10.2e}{np.linalg.norm(final.grad(x)):>10.2e}"
              f"{'' if ok else '  (не сошёлся)'}")


if __name__ == "__main__":
    main()
