import numpy as np


def build_constraints(m, n):
    """Матрица A (m+n, m*n): суммы по строкам и столбцам для x.ravel() (row-major)."""
    A = np.zeros((m + n, m * n))
    for i in range(m):
        A[i, i * n:(i + 1) * n] = 1.0      # sum_j x_ij = s_i
    for j in range(n):
        A[m + j, j::n] = 1.0               # sum_i x_ij = d_j
    return A


class PenaltyOracle:
    """F_mu(x) = f(x) + mu/2 ||Ax - r||^2 + mu/2 sum max(0, -x)^2.  x - плоский вектор длины m*n."""

    def __init__(self, a, b, s, d, mu):
        self.m, self.n = len(s), len(d)
        self.a, self.b = a.ravel(), b.ravel()
        self.A = build_constraints(self.m, self.n)
        self.r = np.concatenate([s, d])
        self.mu = mu

    def f(self, x):
        return self.a @ x + 0.5 * np.sum(self.b * x**2)

    def residual(self, x):
        """Невязка ограничений ||Ax - r|| и нарушение x >= 0."""
        return np.linalg.norm(self.A @ x - self.r), np.linalg.norm(np.minimum(x, 0))

    def func(self, x):
        res = self.A @ x - self.r
        neg = np.minimum(x, 0)
        return self.f(x) + 0.5 * self.mu * (res @ res + neg @ neg)

    def grad(self, x):
        res = self.A @ x - self.r
        neg = np.minimum(x, 0)
        return self.a + self.b * x + self.mu * (self.A.T @ res + neg)

    def hess(self, x):
        # кусочно-постоянный: активна диагональ только там, где x < 0
        return np.diag(self.b) + self.mu * (self.A.T @ self.A + np.diag((x < 0).astype(float)))


def check_grad(oracle, x, eps=1e-6):
    """Центральные конечные разности; возвращает относительную ошибку."""
    g = oracle.grad(x)
    g_fd = np.zeros_like(x)
    for k in range(len(x)):
        e = np.zeros_like(x)
        e[k] = eps
        g_fd[k] = (oracle.func(x + e) - oracle.func(x - e)) / (2 * eps)
    return np.linalg.norm(g - g_fd) / max(np.linalg.norm(g), 1e-12)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    m, n = 3, 4
    a, b = rng.uniform(1, 5, (m, n)), rng.uniform(0.5, 2, (m, n))
    s = rng.uniform(5, 10, m)
    d = rng.uniform(1, 2, n)
    d *= s.sum() / d.sum()                  # баланс sum s = sum d

    for mu in [1, 10, 100, 1000]:
        o = PenaltyOracle(a, b, s, d, mu)
        x = rng.normal(size=m * n)          # есть и отрицательные компоненты
        H = o.hess(x)
        print(f"mu={mu:5}: grad rel.err={check_grad(o, x):.1e}, cond(H)={np.linalg.cond(H):.1e}")
