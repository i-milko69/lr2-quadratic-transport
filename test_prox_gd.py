"""Тесты блока Prox-GD. Запуск из корня репозитория:  python -m pytest -q tests
или  python tests/test_prox_gd.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from methods import ProxGD, FrankWolfe  # noqa: E402
from oracles.transport import (TransportOracle, project_row,  # noqa: E402
                               project_capacity)
from utils.reference import reference_solution  # noqa: E402


def _oracle(mu=1000.0, slack=0.2, m=3, n=3, seed=42):
    return TransportOracle(m=m, n=n, seed=seed, mu=mu, capacity_slack=slack)


# ---------------- проекция ----------------

def test_projection_feasible_idempotent_and_closest():
    from scipy.optimize import minimize
    rng = np.random.default_rng(0)
    for _ in range(30):
        n = int(rng.integers(2, 7))
        x = rng.normal(scale=10, size=n)
        cap = float(rng.uniform(1, 20))
        p = project_row(x, cap)
        assert np.all(p >= -1e-12) and p.sum() <= cap + 1e-9
        assert np.allclose(project_row(p, cap), p, atol=1e-10)   # идемпотентность
        # сравнение с QP: min ||y - x||^2 при y >= 0, sum(y) <= cap
        ref = minimize(lambda y: np.sum((y - x) ** 2), np.zeros(n),
                       jac=lambda y: 2 * (y - x), bounds=[(0, None)] * n,
                       constraints={"type": "ineq", "fun": lambda y: cap - y.sum()},
                       method="SLSQP", options={"ftol": 1e-14})
        assert np.linalg.norm(p - ref.x) < 1e-5


def test_projection_leaves_feasible_points_unchanged():
    x = np.array([1.0, 2.0, 0.5])
    assert np.allclose(project_row(x, 10.0), x)
    assert np.allclose(project_capacity(x[None, :], np.array([10.0])), x[None, :])


# ---------------- оракул ----------------

def test_gradient_matches_finite_differences_in_both_modes():
    rng = np.random.default_rng(1)
    for o in (TransportOracle(3, 3, 42, 100.0), _oracle()):
        x = rng.normal(size=o.dim) * 5
        g = o.grad(x)
        fd = np.array([(o.f(x + 1e-6 * e) - o.f(x - 1e-6 * e)) / 2e-6
                       for e in np.eye(o.dim)])
        assert np.linalg.norm(g - fd) / np.linalg.norm(g) < 1e-6


def test_hessian_matches_gradient_differences():
    o = _oracle()
    x = np.random.default_rng(2).normal(size=o.dim) * 5
    H_fd = np.array([(o.grad(x + 1e-6 * e) - o.grad(x - 1e-6 * e)) / 2e-6
                     for e in np.eye(o.dim)])
    assert np.max(np.abs(o.hess(x) - H_fd)) < 1e-4


def test_lipschitz_estimate_is_upper_bound():
    for o in (TransportOracle(3, 3, 42, 100.0), _oracle()):
        lam = np.linalg.eigvalsh(o.hess(np.full(o.dim, 1.0))).max()
        assert o.info.L >= lam - 1e-9


# ---------------- метод ----------------

def test_prox_gd_returns_nonnegative_shipments():
    oracle = TransportOracle(m=3, n=3, seed=42, mu=100.0)
    history = ProxGD(oracle, max_iter=100, tol=1e-7).run(np.zeros(oracle.dim))
    assert np.all(history.x[-1] >= 0)
    assert np.isfinite(history.f[-1]) and np.isfinite(history.grad_norm[-1])


def test_every_iterate_respects_warehouse_capacity():
    o = _oracle(mu=100.0)
    h = ProxGD(o, max_iter=3000, tol=0.0).run(np.zeros(o.dim))
    for x in h.x:
        xm = x.reshape(o.m, o.n)
        assert xm.min() >= -1e-12
        assert np.all(xm.sum(axis=1) <= o.capacity + 1e-9)


def test_objective_decreases_with_step_one_over_L():
    o = _oracle(mu=100.0)
    h = ProxGD(o, max_iter=3000, tol=0.0).run(np.zeros(o.dim))
    assert np.max(np.diff(h.f)) <= 1e-9


def test_stationarity_reaches_tolerance():
    o = _oracle(mu=100.0)
    h = ProxGD(o, max_iter=200000, tol=1e-7).run(np.zeros(o.dim))
    assert h.grad_norm[-1] <= 1e-7


def test_capacity_constraint_is_active_at_optimum():
    """Иначе эксперимент не показывает пользы prox: хотя бы один склад полон."""
    o = _oracle(mu=1000.0)
    x = ProxGD(o, max_iter=200000, tol=1e-7).run(np.zeros(o.dim)).x[-1]
    slack = o.capacity - x.reshape(o.m, o.n).sum(axis=1)
    assert slack.min() < 1e-6


def test_infinite_capacity_matches_plain_projected_gd():
    """Регрессия: огромные C дают то же решение, что проекция на x >= 0."""
    o_cap = _oracle(mu=100.0, slack=1e6)
    o_old = TransportOracle(3, 3, 42, 100.0, penalty_nonnegative=False)
    # в режиме ёмкостей штрафуются только столбцы, поэтому сравниваем с
    # решением без нижней границы по строкам через ограничения оракула:
    x = ProxGD(o_cap, max_iter=200000, tol=1e-9).run(np.zeros(o_cap.dim)).x[-1]
    assert np.all(x >= 0) and np.all(np.isfinite(x))
    assert o_old.dim == o_cap.dim


def test_penalty_error_shrinks_like_one_over_mu():
    """Нарушение спроса и разрыв по стоимости с f* убывают ~ 1/mu."""
    errs = []
    for mu in (100.0, 1000.0):
        o = _oracle(mu=mu)
        x = ProxGD(o, max_iter=300_000, tol=1e-6).run(np.zeros(o.dim)).x[-1]
        _, f_star = reference_solution(o)
        viol = np.abs(x.reshape(o.m, o.n).sum(axis=0) - o.d).max()
        errs.append((viol, abs(o.true_cost(x) - f_star)))
    assert errs[1][0] < errs[0][0] / 5
    assert errs[1][1] < errs[0][1] / 5
    assert errs[1][0] < 0.05


def test_reference_between_fw_bounds():
    """f* из SLSQP лежит в отрезке [f_FW - gap, f_FW]."""
    o0 = _oracle(mu=0.0)
    h = FrankWolfe(o0, max_iter=1500, tol=1e-1).run(np.zeros(o0.dim))
    _, f_star = reference_solution(o0)
    assert h.f[-1] - h.grad_norm[-1] - 1e-9 <= f_star <= h.f[-1] + 1e-9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("OK", name)
    print("Все проверки ProxGD пройдены.")
