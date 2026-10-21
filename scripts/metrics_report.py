"""Пункт 5: метрики и обусловленность для штрафной транспортной задачи.

Запуск из корня репозитория:
    python -m scripts.metrics_report [--m 3] [--n 3] [--seed 42]

Что считает для каждого mu:
  * cond(H)      - число обусловленности гессиана F_mu в точке x*_mu;
  * ||Ax - r||   - невязка ограничений;
  * f - f*       - разность ИСХОДНОЙ стоимости (без штрафа) и эталона f*;
  * итерации GD  - число шагов до ||grad F_mu|| <= tol;
  * min x        - насколько нарушено x >= 0.
Эталон f* - scipy (SLSQP), sanity check - LP (scipy.linprog) и, если
установлен, cvxpy. Таблица -> figures/metrics_table.csv, графики -> figures/.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog

from methods.gd import GD
from oracles.transport import TransportOracle
from scripts.check_penalty import constraint_matrix, newton_solve, reference_solution
from utils.metrics import condition_number, iterations_to_tol

OUT = "figures"


def f_orig(o, x):
    """Исходная стоимость sum(a x + b x^2 / 2) без штрафных слагаемых."""
    return float(o.a.ravel() @ x + 0.5 * np.sum(o.b.ravel() * x**2))


def reference(o):
    """f* через SLSQP + sanity check через LP (и cvxpy, если есть)."""
    A, r = constraint_matrix(o.m, o.n), np.concatenate([o.s, o.d])
    x_ref, f_ref = reference_solution(o)
    # LP с b=0: нижняя граница, т.к. b >= 0  =>  f_LP* <= f*
    lp = linprog(o.a.ravel(), A_eq=A, b_eq=r, bounds=(0, None), method="highs")
    info = {"f_slsqp": f_ref, "f_lp_lower": lp.fun,
            "viol_slsqp": float(np.abs(A @ x_ref - r).max()),
            "min_x_slsqp": float(x_ref.min())}
    try:
        import cvxpy as cp
        x = cp.Variable(o.dim)
        pb = cp.Problem(cp.Minimize(o.a.ravel() @ x + 0.5 * cp.sum(cp.multiply(o.b.ravel(), cp.square(x)))),
                        [A @ x == r, x >= 0])
        pb.solve()
        info["f_cvxpy"] = float(pb.value)
    except ImportError:
        pass
    assert info["f_lp_lower"] <= f_ref + 1e-8, "LP-граница выше f*: эталон неверен"
    return x_ref, f_ref, info


def run(m, n, seed, mus, tol, max_iter):
    A = constraint_matrix(m, n)
    o0 = TransportOracle(m, n, seed, mu=1.0)
    r = np.concatenate([o0.s, o0.d])
    _, f_star, ref_info = reference(o0)

    rows, hists = [], {}
    for mu in mus:
        o = TransportOracle(m, n, seed, mu=mu)
        x_mu = newton_solve(o, np.zeros(o.dim))            # точный минимум F_mu
        F_mu_star = o.f(x_mu)
        h = GD(o, max_iter=max_iter, tol=tol).run(np.zeros(o.dim))  # шаг 1/L
        it = iterations_to_tol(h, tol)
        x_end = h.x[-1]
        rows.append({
            "mu": mu,
            "cond_H": condition_number(o, x_mu),
            "residual": float(np.linalg.norm(A @ x_mu - r)),
            "f_minus_fstar": f_orig(o, x_mu) - f_star,
            "min_x": float(x_mu.min()),
            "gd_iters": it if it is not None else f">{max_iter}",
            "gd_converged": it is not None,
            "gd_residual_final": float(np.linalg.norm(A @ x_end - r)),
        })
        hists[mu] = np.array(h.f) - F_mu_star
        print(f"mu={mu:8g} cond={rows[-1]['cond_H']:.2e} res={rows[-1]['residual']:.2e} "
              f"f-f*={rows[-1]['f_minus_fstar']:+.2e} GD iters={rows[-1]['gd_iters']}")
    return f_star, ref_info, rows, hists


def make_plots(rows, hists, tag):
    os.makedirs(OUT, exist_ok=True)
    mus = [r["mu"] for r in rows]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(mus, [r["cond_H"] for r in rows], "o-", label="cond(H) в x*_mu")
    ax.set_xlabel("mu"); ax.set_ylabel("cond(H)"); ax.grid(True, which="both", alpha=.3)
    ax.set_title("Обусловленность растёт с mu"); ax.legend()
    fig.savefig(f"{OUT}/cond_vs_mu_{tag}.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(mus, [r["residual"] for r in rows], "o-", label="||Ax - r||")
    ax.loglog(mus, [abs(r["f_minus_fstar"]) for r in rows], "s-", label="|f - f*|")
    ax.set_xlabel("mu"); ax.grid(True, which="both", alpha=.3); ax.legend()
    ax.set_title("Качество решения исходной задачи")
    fig.savefig(f"{OUT}/quality_vs_mu_{tag}.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(mus, [r["gd_iters"] if r["gd_converged"] else np.nan for r in rows], "o-")
    for r in rows:
        if not r["gd_converged"]:
            ax.annotate("не сошёлся", (r["mu"], 1), color="red", fontsize=8)
    ax.set_xlabel("mu"); ax.set_ylabel("итераций GD до ||grad||<=tol")
    ax.grid(True, which="both", alpha=.3); ax.set_title("Число итераций GD (шаг 1/L)")
    fig.savefig(f"{OUT}/iters_vs_mu_{tag}.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for mu, d in hists.items():
        ax.semilogy(np.maximum(d, 1e-16), label=f"mu={mu:g}")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("итерация"); ax.set_ylabel("F_mu(x_k) - F_mu*")
    ax.grid(True, which="both", alpha=.3); ax.legend(); ax.set_title("Сходимость GD")
    fig.savefig(f"{OUT}/convergence_{tag}.png", dpi=150, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=3)
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--max-iter", type=int, default=200000)
    a = p.parse_args()

    mus = [1, 10, 100, 1e3, 1e4]
    f_star, info, rows, hists = run(a.m, a.n, a.seed, mus, a.tol, a.max_iter)
    print(f"\nf* (SLSQP) = {f_star:.8f}; sanity: {info}")

    tag = f"m{a.m}n{a.n}s{a.seed}"
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/metrics_table_{tag}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    make_plots(rows, hists, tag)
    print(f"Таблица и графики сохранены в {OUT}/")
