"""Пункт 5: метрики и обусловленность для штрафной транспортной задачи.

Запуск из корня репозитория:
    python -m scripts.metrics_report [--m 3] [--n 3] [--seed 42]

Что считает для каждого mu:
  * cond(H)      - число обусловленности гессиана F_mu в точке x*_mu;
  * ||Ax - r||   - невязка ограничений;
  * f - f*       - разность ИСХОДНОЙ стоимости (без штрафа) и эталона f*;
  * итерации GD  - число шагов GD (шаг 1/L, старт x=0) по двум критериям:
        gap  : F_mu(x_k) - F_mu* <= eps        (как в exp_conditioning.py, пункт 6)
        grad : ||grad F_mu|| <= tol*||grad F_mu(x0)||   (как в compare_adaptive.py, пункт 7);
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
from utils.metrics import condition_number

OUT = "figures"


def f_orig(o, x):
    """Исходная стоимость sum(a x + b x^2 / 2) без штрафных слагаемых."""
    return float(o.a.ravel() @ x + 0.5 * np.sum(o.b.ravel() * x**2))


def exact_equality_qp(o):
    """Точное решение задачи при Ax = r (без x >= 0) из системы KKT:
        diag(b) x + A^T lam = -a,   A x = r.
    Если все x_ij >= 0, это и есть решение исходной задачи (x >= 0 не активно).
    """
    A, r = constraint_matrix(o.m, o.n), np.concatenate([o.s, o.d])
    k = A.shape[0]
    K = np.block([[np.diag(o.b.ravel()), A.T], [A, np.zeros((k, k))]])
    rhs = np.concatenate([-o.a.ravel(), r])
    x = np.linalg.lstsq(K, rhs, rcond=None)[0][:o.dim]   # A неполного ранга -> lstsq
    return x


def reference(o):
    """f* точно (KKT), SLSQP как перекрёстная проверка, LP как нижняя граница."""
    A, r = constraint_matrix(o.m, o.n), np.concatenate([o.s, o.d])
    x_slsqp, f_slsqp = reference_solution(o)
    x_ref = exact_equality_qp(o)
    if x_ref.min() < -1e-9:           # x >= 0 активно -> KKT неприменим, берём SLSQP
        x_ref, f_ref = x_slsqp, f_slsqp
    else:
        f_ref = f_orig(o, x_ref)
    # LP с b=0: нижняя граница, т.к. b >= 0  =>  f_LP* <= f*
    lp = linprog(o.a.ravel(), A_eq=A, b_eq=r, bounds=(0, None), method="highs")
    info = {"f_exact": f_ref, "f_slsqp": f_slsqp,
            "slsqp_error": float(f_slsqp - f_ref),     # у SLSQP может быть недосходимость
            "f_lp_lower": lp.fun,
            "viol_exact": float(np.abs(A @ x_ref - r).max()),
            "min_x_exact": float(x_ref.min())}
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


def gd_iterations(o, F_star, eps_gap, rel_tol, max_iter, chunk=5000):
    """Один прогон GD (шаг 1/L, x0 = 0), число итераций по двум критериям.

    gap : первая итерация с F(x_k) - F_star <= eps_gap
    grad: первая итерация с ||grad F(x_k)|| <= rel_tol * ||grad F(x_0)||
    Идём кусками по chunk итераций и останавливаемся, когда достигнуты оба.
    Возвращает (it_gap, it_grad, f_history, x_last); None = критерий не достигнут.
    """
    x = np.zeros(o.dim)
    tol_abs = rel_tol * float(np.linalg.norm(o.grad(x)))
    f_all, g_all, total = [], [], 0
    it_gap = it_grad = None
    while total < max_iter and (it_gap is None or it_grad is None):
        n = min(chunk, max_iter - total)
        h = GD(o, max_iter=n, tol=0.0).run(x)           # tol=0: не останавливаться раньше
        start = 0 if total == 0 else 1                  # индекс 0 куска = последняя точка прошлого
        f_all += h.f[start:]
        g_all += h.grad_norm[start:]
        x, total = h.x[-1], total + n
        if it_gap is None:
            hit = np.flatnonzero(np.array(f_all) - F_star <= eps_gap)
            it_gap = int(hit[0]) if hit.size else None
        if it_grad is None:
            hit = np.flatnonzero(np.array(g_all) <= tol_abs)
            it_grad = int(hit[0]) if hit.size else None
    return it_gap, it_grad, f_all, x


def run(m, n, seed, mus, tol, eps_gap, max_iter):
    A = constraint_matrix(m, n)
    o0 = TransportOracle(m, n, seed, mu=1.0)
    r = np.concatenate([o0.s, o0.d])
    _, f_star, ref_info = reference(o0)

    rows, hists = [], {}
    for mu in mus:
        o = TransportOracle(m, n, seed, mu=mu)
        x_mu = newton_solve(o, np.zeros(o.dim))            # точный минимум F_mu
        F_mu_star = o.f(x_mu)
        it_gap, it_grad, f_hist, x_end = gd_iterations(o, F_mu_star, eps_gap, tol, max_iter)
        fmt = lambda it: it if it is not None else f">{max_iter}"
        rows.append({
            "mu": mu,
            "cond_H": condition_number(o, x_mu),
            "residual": float(np.linalg.norm(A @ x_mu - r)),
            "f_minus_fstar": f_orig(o, x_mu) - f_star,
            "min_x": float(x_mu.min()),
            "gd_iters_gap": fmt(it_gap),
            "gd_iters_grad": fmt(it_grad),
            "gd_converged_gap": it_gap is not None,
            "gd_converged_grad": it_grad is not None,
            "gd_residual_final": float(np.linalg.norm(A @ x_end - r)),
        })
        hists[mu] = np.array(f_hist) - F_mu_star
        print(f"mu={mu:8g} cond={rows[-1]['cond_H']:.2e} res={rows[-1]['residual']:.2e} "
              f"f-f*={rows[-1]['f_minus_fstar']:+.2e} "
              f"GD iters: gap={rows[-1]['gd_iters_gap']} grad={rows[-1]['gd_iters_grad']}")
    return f_star, ref_info, rows, hists


def make_plots(rows, hists, tag, crit="gap"):
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
    ax.loglog(mus, [r[f"gd_iters_{crit}"] if r[f"gd_converged_{crit}"] else np.nan for r in rows], "o-")
    for r in rows:
        if not r[f"gd_converged_{crit}"]:
            ax.annotate("не сошёлся", (r["mu"], 1), color="red", fontsize=8)
    ax.set_xlabel("mu"); ax.set_ylabel("итераций GD: F-F* <= eps" if crit == "gap" else "итераций GD: ||grad|| <= tol*||grad(x0)||")
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
    p.add_argument("--tol", type=float, default=1e-6)        # критерий grad: ||grad|| <= tol*||grad(x0)||
    p.add_argument("--eps-gap", type=float, default=1e-4)    # критерий gap: F_mu(x_k) - F_mu* <= eps
    p.add_argument("--criterion", choices=["gap", "grad"], default="gap")  # что рисовать на графике итераций
    p.add_argument("--max-iter", type=int, default=300000)   # предел итераций GD
    a = p.parse_args()

    mus = [1, 10, 100, 1e3, 1e4]
    f_star, info, rows, hists = run(a.m, a.n, a.seed, mus, a.tol, a.eps_gap, a.max_iter)
    print(f"\nf* (точное, KKT) = {f_star:.8f}; проверки: {info}")

    tag = f"m{a.m}n{a.n}s{a.seed}"
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/metrics_table_{tag}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    make_plots(rows, hists, tag, a.criterion)
    print(f"Таблица и графики сохранены в {OUT}/")
