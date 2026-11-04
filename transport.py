
import numpy as np
from .base import Oracle, OracleInfo


def generate_data(m=3, n=3, seed=42):
    """Генерирует синтетические данные для транспортной задачи.

    m — количество складов.
    n — количество магазинов.
    seed — начальное значение для генератора случайных чисел.
    """
    rng = np.random.default_rng(seed)

    # Генерируем тарифы перевозки и коэффициенты квадратичных затрат.
    a = rng.uniform(1.0, 10.0, size=(m, n))
    b = rng.uniform(0.1, 2.0, size=(m, n))

    # Генерируем запасы складов и спрос магазинов.
    supply_raw = rng.uniform(10, 100, size=m)
    demand_raw = rng.uniform(10, 100, size=n)

    # Балансируем задачу: общий запас равен общему спросу.
    total = 100.0
    s = supply_raw * (total / supply_raw.sum())
    d = demand_raw * (total / demand_raw.sum())

    return a, b, s, d


def project_row(x, cap):
    """Евклидова проекция вектора x на {y >= 0, sum(y) <= cap}."""
    y = np.maximum(x, 0.0)
    if y.sum() <= cap:
        return y
    # Иначе ограничение активно: проекция на симплекс sum(y) = cap, y >= 0.
    u = np.sort(x)[::-1]
    css = np.cumsum(u) - cap
    k = np.arange(1, len(x) + 1)
    rho = np.nonzero(u - css / k > 0)[0][-1]
    theta = css[rho] / (rho + 1)
    return np.maximum(x - theta, 0.0)


def project_capacity(x, capacity):
    """Построчная проекция матрицы x (m x n) на {x >= 0, sum_j x_ij <= C_i}."""
    x = np.asarray(x, dtype=float)
    return np.vstack([project_row(x[i], capacity[i]) for i in range(len(capacity))])


def f(x, a, b, s, d, mu=100.0, penalty_nonnegative=True,
      penalty_rows_on=True):
    """Вычисляет целевую функцию транспортной задачи со штрафами."""
    x = np.asarray(x).reshape(a.shape)

    # Считаем исходную стоимость перевозок.
    f_orig = np.sum(a * x + 0.5 * b * x**2)

    # Считаем отклонения от запасов складов и спроса магазинов.
    row_diff = x.sum(axis=1) - s
    col_diff = x.sum(axis=0) - d

    # Штрафуем за нарушение баланса и отрицательные перевозки.
    penalty_rows = np.sum(row_diff**2)
    penalty_cols = np.sum(col_diff**2)
    penalty_neg = np.sum(np.maximum(0.0, -x) ** 2)

    if not penalty_nonnegative:
        penalty_neg = 0.0
    if not penalty_rows_on:
        penalty_rows = 0.0

    return f_orig + 0.5 * mu * (
        penalty_rows + penalty_cols + penalty_neg
    )


def grad(x, a, b, s, d, mu=100.0, penalty_nonnegative=True,
         penalty_rows_on=True):
    """Вычисляет градиент целевой функции со штрафами."""
    x = np.asarray(x).reshape(a.shape)

    # Градиент исходной стоимости перевозок.
    grad_orig = a + b * x

    # Отклонения объёмов перевозок от запасов и спроса.
    row_diff = x.sum(axis=1) - s
    col_diff = x.sum(axis=0) - d

    # Градиенты штрафов за нарушение баланса.
    grad_rows = np.broadcast_to(row_diff[:, None], x.shape)
    if not penalty_rows_on:
        grad_rows = np.zeros_like(x)
    grad_cols = np.broadcast_to(col_diff[None, :], x.shape)

    # Градиент штрафа за отрицательные объёмы перевозок.
    grad_neg = np.minimum(x, 0.0)
    if not penalty_nonnegative:
        grad_neg = np.zeros_like(x)

    # Возвращаем сумму градиентов исходной функции и штрафов.
    return (
        grad_orig 
        + mu * (grad_rows + grad_cols + grad_neg)
    )


class TransportOracle(Oracle):
    """Оракул транспортной задачи.

    Два режима.

    1. capacity_slack=None (по умолчанию) — классическая постановка:
       sum_j x_ij = s_i, sum_i x_ij = d_j, x >= 0; все ограничения
       учитываются квадратичным штрафом с весом mu.

    2. capacity_slack=t (например, 0.2) — ограничения по объёму склада:
       sum_j x_ij <= C_i, C_i = s_i (1 + t); спрос по-прежнему равенство
       sum_i x_ij = d_j (штраф), а x >= 0 и ёмкости склада учитываются
       ТОЧНО через проекцию prox (штрафов за строки и x < 0 нет).
       Этот режим предназначен для ProxGD и Франка-Вульфа.
    """

    def __init__(self, m=3, n=3, seed=42, mu=100.0, penalty_nonnegative=True,
                 capacity_slack=None):
        self.a, self.b, self.s, self.d = generate_data(m=m, n=n, seed=seed)

        self.m = m
        self.n = n
        self.penalty = float(mu)
        self.penalty_nonnegative = bool(penalty_nonnegative)
        self.dim = m * n
        self.name = "transport"

        self.capacity_slack = capacity_slack
        if capacity_slack is None:
            self.capacity = None
            self._rows_penalized = True
        else:
            if capacity_slack < 0:
                raise ValueError("capacity_slack должен быть >= 0")
            # sum C_i >= sum d_j выполнено, т.к. sum s = sum d.
            self.capacity = self.s * (1.0 + float(capacity_slack))
            self._rows_penalized = False
            # Неотрицательность обеспечивает prox, штраф не нужен.
            self.penalty_nonnegative = False

        # Константа Липшица градиента (верхняя оценка) для выбора шага.
        # Строки штрафуются: блок строк даёт mu*n, блок столбцов mu*m.
        k = (self.m + self.n) if self._rows_penalized else self.m
        self._L = float(np.max(self.b)) + self.penalty * (
            k + (1 if self.penalty_nonnegative else 0))

    def _kw(self):
        return dict(mu=self.penalty,
                    penalty_nonnegative=self.penalty_nonnegative,
                    penalty_rows_on=self._rows_penalized)

    def f(self, x):
        return float(f(x, self.a, self.b, self.s, self.d, **self._kw()))

    def grad(self, x):
        return grad(x, self.a, self.b, self.s, self.d, **self._kw()).ravel()

    def prox(self, x, step=None):
        """Проекция на {x >= 0} (и sum_j x_ij <= C_i, если заданы ёмкости).

        Для индикаторной функции множества prox = проекция и не зависит
        от шага.
        """
        x_mat = np.asarray(x, dtype=float).reshape(self.m, self.n)
        if self.capacity is None:
            return np.maximum(x_mat, 0.0).ravel()
        return project_capacity(x_mat, self.capacity).ravel()

    def true_cost(self, x):
        """Исходная стоимость sum(a x + b x^2 / 2) без штрафов."""
        x = np.asarray(x).reshape(self.a.shape)
        return float(np.sum(self.a * x + 0.5 * self.b * x**2))

    def hess(self, x):
        """Матрица Гессе целевой функции (со штрафами)."""
        x_mat = np.asarray(x).reshape(self.m, self.n)
        H = np.diag(self.b.ravel())

        if self._rows_penalized:
            for i in range(self.m):
                idx = np.arange(i * self.n, (i + 1) * self.n)
                H[np.ix_(idx, idx)] += self.penalty

        for j in range(self.n):
            idx = np.arange(j, self.m * self.n, self.n)
            H[np.ix_(idx, idx)] += self.penalty

        if self.penalty_nonnegative:
            H[np.diag_indices(self.dim)] += (
                self.penalty * (x_mat.ravel() < 0)
            )
        return H

    def hess_diag(self, x):
        """Диагональ гессиана (для PreconditionedGD)."""
        x = np.asarray(x).ravel()
        rows = self.penalty if self._rows_penalized else 0.0
        neg = self.penalty * (x < 0) if self.penalty_nonnegative else 0.0
        return self.b.ravel() + rows + self.penalty + neg

    @property
    def info(self):
        return OracleInfo(L=self._L)
