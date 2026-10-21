"""Эксперимент пункта 6: обусловленность и число итераций GD.

Запуск из корня репозитория:  python exp_conditioning.py
Сравниваем варианты при растущем mu:
  gd            — baseline (methods.GD, шаг 1/L)
  scale_b       — замена x -> x/sqrt(b)
  jacobi        — диагональное предобуславливание b + 2*mu
  structured    — предобуславливатель (diag(b) + mu A^T A)^-1
  cont          — continuation по mu (рост x10, тёплый старт), внутри plain GD,
                  промежуточные этапы решаются грубо (EPS_MID)
  cont_tight    — то же, но промежуточные этапы решаются до той же точности EPS
  struct_active — continuation + структурный предобуславливатель с активным
                  множеством x<0 из предыдущего этапа (шаг 1/L, L ~ 1)
"""
import numpy as np
from scipy.optimize import minimize

from oracles.transport import TransportOracle
from conditioning import (constraint_matrix, set_mu, set_b, cond, scale_by_b,
                          jacobi, structured, newton_min, iters_to_gap)

M = N = 3
SEED = 42
EPS = 1e-4          # допуск по F_mu(x) - F_mu* на целевом mu
EPS_MID = 1e-1      # допуск на промежуточных этапах continuation
MAX_ITER = 300_000


def make(mu, wide_b):
    o = TransportOracle(M, N, SEED, mu)
    if wide_b:  # логравномерно от 1e-2 до 1e2: большой разброс масштабов
        rng = np.random.default_rng(SEED + 1)
        set_b(o, 10 ** rng.uniform(-2, 2, size=(M, N)))
    return o


def reference(o):
    """Эталон исходной задачи (с ограничениями): f*, x* через SLSQP."""
    A = constraint_matrix(o.m, o.n)
    r = np.concatenate([o.s, o.d])
    A, r = A[:-1], r[:-1]                     # баланс => одно ограничение лишнее
    fo = lambda x: np.sum(o.a.ravel() * x + 0.5 * o.b.ravel() * x**2)
    go = lambda x: o.a.ravel() + o.b.ravel() * x
    res = minimize(fo, np.full(o.dim, 100.0 / o.dim), jac=go, method="SLSQP",
                   bounds=[(0, None)] * o.dim,
                   constraints=[{"type": "eq", "fun": lambda x: A @ x - r,
                                 "jac": lambda x: A}],
                   options={"maxiter": 500, "ftol": 1e-14})
    return res.x, fo(res.x)


def solve_stage(o, wrapper, x0, eps):
    """Один запуск GD для текущего mu. Возвращает x, итерации, ok."""
    ow = wrapper(o) if wrapper else o
    F_star = o.f(newton_min(o, x0))
    y0 = ow.to_y(x0) if wrapper else x0
    y, it, ok = iters_to_gap(ow, y0, F_star, eps, MAX_ITER)
    return (ow.to_x(y) if wrapper else y), it, ok


def run_struct_active(mu, wide_b, max_refresh=5, inner=200):
    """Continuation + structured(x_ref): x_ref — решение предыдущего этапа.
    Если активное множество угадано неверно и GD не сошёлся за inner итераций,
    пересобираем предобуславливатель по текущей точке (до max_refresh раз)."""
    x, total, k, x_ref, ok = np.zeros(M * N), 0, 1.0, None, True
    while k < mu * 1.0001:
        o = make(k, wide_b)
        eps = EPS if k >= mu / 1.0001 else EPS_MID
        F_star = o.f(newton_min(o, x))
        ref = x_ref
        for _ in range(max_refresh):
            ow = structured(o, ref)
            y, it, ok = iters_to_gap(ow, ow.to_y(x), F_star, eps, inner, chunk=inner)
            total += it
            x = ow.to_x(y)
            if ok:
                break
            ref = x
        x_ref = x
        k *= 10
    return make(mu, wide_b), x, total, ok


def run(variant, mu, wide_b):
    x0 = np.zeros(M * N)
    if variant == "struct_active":
        return run_struct_active(mu, wide_b)
    if variant in ("cont", "cont_tight"):
        eps_mid = EPS_MID if variant == "cont" else EPS
        total, x, ok, k = 0, x0, True, 1.0
        while k < mu * 1.0001:
            o = make(k, wide_b)
            x, it, ok = solve_stage(o, None, x, EPS if k >= mu / 1.0001 else eps_mid)
            total += it
            k *= 10
        return make(mu, wide_b), x, total, ok
    wrapper = {"gd": None, "scale_b": scale_by_b, "jacobi": jacobi,
               "structured": structured}[variant]
    o = make(mu, wide_b)
    x, it, ok = solve_stage(o, wrapper, x0, EPS)
    return o, x, it, ok


def main():
    variants = ["gd", "scale_b", "jacobi", "structured", "cont", "cont_tight", "struct_active"]
    for wide_b in (False, True):
        print(f"\n=== b {'логравномерно [1e-2,1e2]' if wide_b else 'U(0.1,2) как в generate_data'} ===")
        ref_o = make(1.0, wide_b)
        x_star, f_star = reference(ref_o)
        A = constraint_matrix(M, N)
        r = np.concatenate([ref_o.s, ref_o.d])
        print(f"f* = {f_star:.4f}")
        print(f"{'mu':>7} {'вариант':<12} {'cond':>10} {'итер':>9} {'f-f*':>10} "
              f"{'|Ax-r|':>9} {'|x-x*|':>9}")
        for mu in (1, 10, 100, 1000, 10000):
            for v in variants:
                if mu == 10000 and v != "struct_active":
                    continue  # plain GD: ~1e6+ итераций, не ждём
                o, x, it, ok = run(v, mu, wide_b)
                xm = newton_min(o, x)  # для cond берём точку минимума F_mu
                base_o = o
                if v in ("scale_b", "jacobi", "structured"):
                    ow = {"scale_b": scale_by_b, "jacobi": jacobi,
                          "structured": structured}[v](o)
                    c = cond(ow, ow.to_y(xm))
                elif v == "struct_active":
                    ow = structured(o, xm)   # активное множество в решении
                    c = cond(ow, ow.to_y(xm))
                else:
                    c = cond(o, xm)
                fo = np.sum(base_o.a.ravel() * x + 0.5 * base_o.b.ravel() * x**2)
                tag = "" if ok else "*"
                print(f"{mu:>7} {v:<12} {c:>10.3g} {it:>8d}{tag or ' '} "
                      f"{fo - f_star:>10.3g} {np.linalg.norm(A @ x - r):>9.2g} "
                      f"{np.linalg.norm(x - x_star):>9.2g}")
            print()
    print("* — точность не достигнута за MAX_ITER")


if __name__ == "__main__":
    main()
