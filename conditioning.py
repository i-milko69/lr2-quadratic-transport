"""Пункт 6: улучшение обусловленности штрафной транспортной задачи.

Гессиан F_mu:  H(x) = diag(b) + mu * A^T A + mu * diag(x < 0).
Положить в корень репозитория (или в utils/); импорты ниже — от корня.
Всё работает поверх существующего TransportOracle, его код не меняется.
"""
import numpy as np
from oracles.base import Oracle, OracleInfo
from oracles.transport import TransportOracle
from methods import GD


# ---------- вспомогательное ----------------------------------------------

def constraint_matrix(m, n):
    """A размера (m+n, m*n): суммы по строкам и столбцам (x хранится построчно)."""
    rows = np.kron(np.eye(m), np.ones((1, n)))
    cols = np.kron(np.ones((1, m)), np.eye(n))
    return np.vstack([rows, cols])


def set_mu(oracle, mu):
    """Сменить штраф у готового оракула (в TransportOracle L считается в __init__)."""
    oracle.penalty = float(mu)
    oracle._L = float(oracle.b.max()) + oracle.penalty * (oracle.m + oracle.n + 1)
    return oracle


def set_b(oracle, b):
    """Подменить b (чтобы получить задачу с большим разбросом масштабов)."""
    oracle.b = np.asarray(b, dtype=float).reshape(oracle.m, oracle.n)
    return set_mu(oracle, oracle.penalty)


def cond(oracle, x):
    """Число обусловленности (обобщённого) гессиана в точке x."""
    ev = np.linalg.eigvalsh(oracle.hess(x))
    return float(ev[-1] / ev[0])


# ---------- замена переменных x = T y (масштабирование / предобуславливание)

class ScaledOracle(Oracle):
    """F~(y) = F_mu(T y).  GD на F~ по y  <=>  x <- x - eta * (T T^T) grad F(x).

    T — вектор t (диагональная матрица) или плотная матрица.
      * масштабирование из плана:   t = 1/sqrt(b)           (y = sqrt(b) * x)
      * диагональный Якоби:         t = 1/sqrt(b + 2*mu)
      * структурный предобуславл.:  T = chol((diag(b) + mu A^T A)^-1)
    Метрики качества считать в исходных x: oracle.to_x(y).
    """

    def __init__(self, base: TransportOracle, T, neg_mask=None):
        """neg_mask — для каких координат в оценке L учитывать штраф за x < 0.
        None = для всех (безопасная верхняя оценка, как в TransportOracle)."""
        self.base = base
        self.T = np.asarray(T, dtype=float)
        self.diag = self.T.ndim == 1
        self.dim = base.dim
        self.name = f"scaled({base.name})"
        # L — по верхней оценке Гессиана  H <= diag(b) + mu*(A^T A + I)
        A = constraint_matrix(base.m, base.n)
        mask = np.ones(self.dim) if neg_mask is None else np.asarray(neg_mask, float)
        Hmax = np.diag(base.b.ravel()) + base.penalty * (A.T @ A + np.diag(mask))
        Tm = np.diag(self.T) if self.diag else self.T
        self._L = float(np.linalg.eigvalsh(Tm.T @ Hmax @ Tm)[-1])

    def to_x(self, y):
        return self.T * y if self.diag else self.T @ y

    def to_y(self, x):
        return x / self.T if self.diag else np.linalg.solve(self.T, x)

    def f(self, y):
        return self.base.f(self.to_x(y))

    def grad(self, y):
        g = self.base.grad(self.to_x(y))
        return self.T * g if self.diag else self.T.T @ g

    def hess(self, y):
        H = self.base.hess(self.to_x(y))
        if self.diag:
            return H * np.outer(self.T, self.T)
        return self.T.T @ H @ self.T

    @property
    def info(self):
        return OracleInfo(L=self._L)


def scale_by_b(base):
    """Замена x -> x / sqrt(b): квадратичная часть становится 1/2 ||y||^2."""
    return ScaledOracle(base, 1.0 / np.sqrt(base.b.ravel()))


def jacobi(base):
    """Диагональное предобуславливание diag(H) = b + 2*mu."""
    return ScaledOracle(base, 1.0 / np.sqrt(base.b.ravel() + 2.0 * base.penalty))


def structured(base, x_ref=None):
    """Предобуславливатель (diag(b') + mu A^T A)^-1, b' = b + mu*[x_ref < 0].

    Учитывает структуру A и (если задан x_ref, например решение предыдущего
    этапа continuation) активные ограничения x >= 0: у них штраф даёт
    дополнительную кривизну mu. Через Вудбери обращение сводится к системе
    размера (m+n); здесь для простоты считаем плотно (dim = m*n небольшой).
    """
    A = constraint_matrix(base.m, base.n)
    d = base.b.ravel().copy()
    if x_ref is not None:
        d += base.penalty * (np.asarray(x_ref) < 0)
    H0 = np.diag(d) + base.penalty * (A.T @ A)
    mask = None if x_ref is None else (np.asarray(x_ref) < 0)
    return ScaledOracle(base, np.linalg.cholesky(np.linalg.inv(H0)), neg_mask=mask)


# ---------- эталоны и baseline-GD ------------------------------------------

def newton_min(oracle, x0, tol=1e-11, max_iter=200):
    """Точный минимум F_mu (демпфированный Ньютон) — для измерения f - f*."""
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


def iters_to_gap(oracle, y0, F_star, eps, max_iter=300_000, chunk=2000):
    """Запускает GD из methods/ (шаг 1/L) и возвращает первую итерацию, на
    которой F(y) - F_star <= eps.

    Критерий по разности значений функции не зависит от замены переменных,
    поэтому честно сравним между вариантами (а ||grad|| в y и в x — разные).
    Идём кусками по chunk итераций, чтобы не гонять лишнее после попадания.
    Возвращает (y, число итераций, достигнута ли точность).
    """
    y, total = np.array(y0, dtype=float), 0
    while total < max_iter:
        n = min(chunk, max_iter - total)
        h = GD(oracle, max_iter=n, tol=0.0).run(y)
        hit = np.flatnonzero(np.array(h.f) - F_star <= eps)
        if hit.size:
            return h.x[hit[0]], total + int(hit[0]), True
        y, total = h.x[-1], total + n
    return y, total, False


def wrap(oracle, kind):
    """Для YAML-конфига: scaling: scale_b | jacobi | structured."""
    return {"scale_b": scale_by_b, "jacobi": jacobi, "structured": structured}[kind](oracle)
