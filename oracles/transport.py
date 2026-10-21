
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


def f(x, a, b, s, d, mu=100.0, penalty_nonnegative=True):
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

    return f_orig + 0.5 * mu * (
        penalty_rows + penalty_cols + penalty_neg
    )


def grad(x, a, b, s, d, mu=100.0, penalty_nonnegative=True):
    """Вычисляет градиент целевой функции со штрафами."""
    x = np.asarray(x).reshape(a.shape)

    # Градиент исходной стоимости перевозок.
    grad_orig = a + b * x

    # Отклонения объёмов перевозок от запасов и спроса.
    row_diff = x.sum(axis=1) - s
    col_diff = x.sum(axis=0) - d

    # Градиенты штрафов за нарушение баланса.
    grad_rows = np.broadcast_to(row_diff[:, None], x.shape)
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
    """Оракул транспортной задачи с квадратичными штрафами."""

    def __init__(self, m=3, n=3, seed=42, mu=100.0, penalty_nonnegative=True):
        # Создаём данные транспортной задачи.
        self.a, self.b, self.s, self.d = generate_data(
            m=m, n=n, seed=seed
        )

        self.m = m
        self.n = n
        self.penalty = float(mu)
        self.penalty_nonnegative = bool(penalty_nonnegative)
        self.dim = m * n
        self.name = "transport"

        # Оцениваем константу Липшица градиента.
        # Она используется для выбора шага градиентного спуска.
        self._L = float(np.max(self.b)) + self.penalty * (m + n + 1)

    def f(self, x):
        return float(
            f(
                x, self.a, self.b, self.s, self.d,
                self.penalty, self.penalty_nonnegative
            )
        )

    def grad(self, x):
        return grad(
            x, self.a, self.b, self.s, self.d,
            self.penalty, self.penalty_nonnegative
        ).ravel()

    def hess(self, x):
        """Вычисляет матрицу Гессе целевой функции."""
        x_mat = np.asarray(x).reshape(self.m, self.n)

        # Начинаем с Гессиана исходной квадратичной стоимости.
        H = np.diag(self.b.ravel())

        # Добавляем вклад штрафов за нарушение запасов складов.
        for i in range(self.m):
            idx = np.arange(i * self.n, (i + 1) * self.n)
            H[np.ix_(idx, idx)] += self.penalty

        # Добавляем вклад штрафов за нарушение спроса магазинов.
        for j in range(self.n):
            idx = np.arange(j, self.m * self.n, self.n)
            H[np.ix_(idx, idx)] += self.penalty

        # Добавляем вклад штрафа там, где перевозки отрицательны.
        H[np.diag_indices(self.dim)] += (
            self.penalty * (x_mat.ravel() < 0)
        )

        return H

    @property
    def info(self):
        """Возвращает параметры, необходимые методу оптимизации."""
        return OracleInfo(L=self._L)
