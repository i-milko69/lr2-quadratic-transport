
import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from oracles.transport import TransportOracle  # noqa: E402


def constraint_matrix(m, n):
    """A: суммы по строкам и столбцам для x.ravel() (row-major)."""
    A = np.zeros((m + n, m * n))
    for i in range(m):
        A[i, i * n:(i + 1) * n] = 1.0
    for j in range(n):
        A[m + j, j::n] = 1.0
    return A


def fd_grad_error(o, x, eps=1e-6):
    """Относительная ошибка аналитического градиента против центральных разностей."""
    g = o.grad(x)
    g_fd = np.array([(o.f(x + eps * e) - o.f(x - eps * e)) / (2 * eps) for e in np.eye(len(x))])
    return np.linalg.norm(g - g_fd) / np.linalg.norm(g)


def fd_hess_error(o, x, eps=1e-6):
    """Гессиан против конечных разностей градиента."""
    H = o.hess(x)
    H_fd = np.array([(o.grad(x + eps * e) - o.grad(x - eps * e)) / (2 * eps) for e in np.eye(len(x))])
    return np.linalg.norm(H - H_fd) / np.linalg.norm(H)


def newton_solve(o, x0, iters=100, tol=1e-10):
    """Точный минимизатор F_mu (кусочно-квадратичная, C^1) методом Ньютона."""
    x = x0.copy()
    for _ in range(iters):
        g = o.grad(x)
        if np.linalg.norm(g) < tol:
            break
        x = x - np.linalg.solve(o.hess(x), g)
    return x

def reference_solution_exact(o):
    A, r = constraint_matrix(o.m, o.n), np.concatenate([o.s, o.d])
    a, b = o.a.ravel(), o.b.ravel()
    K = np.block([[np.diag(b), A.T], [A, np.zeros((len(r), len(r)))]])
    x = np.linalg.lstsq(K, np.concatenate([-a, r]), rcond=None)[0][:o.dim]
    if x.min() < 0:
        return None
    return x, a @ x + 0.5 * np.sum(b * x**2)
    
def reference_solution(o):
    """f* исходной задачи (с ограничениями) через SLSQP."""
        exact = reference_solution_exact(o)
    if exact is not None:
        return exact
    A, r = constraint_matrix(o.m, o.n), np.concatenate([o.s, o.d])
    a, b = o.a.ravel(), o.b.ravel()
    res = minimize(lambda x: a @ x + 0.5 * np.sum(b * x**2), np.full(o.dim, r[:o.m].sum() / o.dim),
                   jac=lambda x: a + b * x, bounds=[(0, None)] * o.dim,
                   constraints={"type": "eq", "fun": lambda x: A @ x - r, "jac": lambda x: A},
                   method="SLSQP", options={"ftol": 1e-12, "maxiter": 500})
    return res.x, res.fun


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    o0 = TransportOracle(m=3, n=3, seed=42, mu=100.0)
    x_ref, f_ref = reference_solution(o0)
    A, r = constraint_matrix(o0.m, o0.n), np.concatenate([o0.s, o0.d])
    print(f"Эталон f* = {f_ref:.6f}\n")
    print(f"{'mu':>8} {'grad err':>10} {'hess err':>10} {'cond(H*)':>10} "
          f"{'||Ax-r||':>10} {'f(x*)-f*':>11} {'L (info)':>10} {'L (true)':>10}")
    for mu in [1, 10, 100, 1e3, 1e4]:
        o = TransportOracle(m=3, n=3, seed=42, mu=mu)
        x_test = rng.normal(size=o.dim) * 5          # есть отрицательные компоненты
        x_star = newton_solve(o, np.zeros(o.dim))
        H_star = o.hess(x_star)
        print(f"{mu:8.0f} {fd_grad_error(o, x_test):10.1e} {fd_hess_error(o, x_test):10.1e} "
              f"{np.linalg.cond(H_star):10.1e} {np.linalg.norm(A @ x_star - r):10.2e} "
              f"{o.a.ravel() @ x_star + 0.5 * np.sum(o.b.ravel() * x_star**2) - f_ref:11.2e} "
              f"{o.info.L:10.1e} {np.linalg.eigvalsh(H_star).max():10.1e}")
