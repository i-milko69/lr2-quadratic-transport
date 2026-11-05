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
from scipy.optimize import linprog, minimize

from methods.gd import GD
from oracles.transport import TransportOracle
from scripts.check_penalty import constraint_matrix, newton_solve
from utils.metrics import condition_number

OUT = "figures"


def f_orig(o, x):
    """Исходная стоимость sum(a x + b x^2 / 2) без штрафных слагаемых."""
    return float(o.a.ravel() @ x + 0.5 * np.sum(o.b.ravel() * x**2))


def scipy_reference(o):
    """Эталон f*: scipy.optimize.minimize (SLSQP) на ИСХОДНОЙ задаче с ограничениями.

    Важно: уравнений баланса m+n, но независимых только m+n-1 (sum s = sum d),
    а SLSQP плохо переносит линейно зависимые ограничения (без этого он
    останавливается с ошибкой 0.02-5 по f). Поэтому убираем последнюю строку.
    """
    A, r = constraint_matrix(o.m, o.n), np.concatenate([o.s, o.d])
    Ae, re = A[:-1], r[:-1]
    a, b = o.a.ravel(), o.b.ravel()
    res = minimize(lambda x: a @ x + 0.5 * np.sum(b * x**2),
                   np.full(o.dim, r[:o.m].sum() / o.dim),
                   jac=lambda x: a + b * x, method="SLSQP",
                   bounds=[(0, None)] * o.dim,
                   constraints={"type": "eq", "fun": lambda x: Ae @ x - re, "jac": lambda x: Ae},
                   options={"ftol": 1e-14, "maxiter": 1000})
    assert res.success, f"SLSQP не сошёлся: {res.message}"
    return res.x, float(res.fun)


def cvxpy_reference(o):
    """Тот же f* через cvxpy; None, если cvxpy не установлен."""
    try:
        import cvxpy as cp
    except ImportError:
        return None
    A, r = constraint_matrix(o.m, o.n), np.concatenate([o.s, o.d])
    x = cp.Variable(o.dim)
    obj = o.a.ravel() @ x + 0.5 * cp.sum(cp.multiply(o.b.ravel(), cp.square(x)))
    pb = cp.Problem(cp.Minimize(obj), [A @ x == r, x >= 0])
    pb.solve()
    return float(pb.value)


def reference(o):
    """f*: scipy (основной), cvxpy (если есть) и LP (sanity check, нижняя граница)."""
    A, r = constraint_matrix(o.m, o.n), np.concatenate([o.s, o.d])
    x_ref, f_ref = scipy_reference(o)
    # sanity check: LP при b = 0. Так как b >= 0, стоимость LP не больше f*.
    lp = linprog(o.a.ravel(), A_eq=A, b_eq=r, bounds=(0, None), method="highs")
    info = {"f_scipy": f_ref, "f_lp_lower": lp.fun,
            "viol_scipy": float(np.abs(A @ x_ref - r).max()),
            "min_x_scipy": float(x_ref.min())}
    f_cvx = cvxpy_reference(o)
    if f_cvx is not None:
        info["f_cvxpy"] = f_cvx
        assert abs(f_cvx - f_ref) < 1e-4 * max(1.0, abs(f_ref)), "scipy и cvxpy не совпали"
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

    rows, hists, devs = [], {}, {}
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
        devs[mu] = A @ x_mu - r              # недовоз/перевоз по каждому складу и магазину
        print(f"mu={mu:8g} cond={rows[-1]['cond_H']:.2e} res={rows[-1]['residual']:.2e} "
              f"f-f*={rows[-1]['f_minus_fstar']:+.2e} "
              f"GD iters: gap={rows[-1]['gd_iters_gap']} grad={rows[-1]['gd_iters_grad']}")
    return f_star, ref_info, rows, hists, devs


def _num(v):
    """'>300000' -> 300000 (для нерасшедшихся точек), число остаётся числом."""
    return int(str(v).lstrip(">")) if isinstance(v, str) else v


def _save(fig, name, tag):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/{name}_{tag}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _setup(ax, title, xlabel="μ (коэффициент штрафа)", ylabel=None):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(True, which="both", alpha=.25)
    ax.set_title(title, fontsize=11); ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def _ref(ax, mus, ys, power, label):
    """Пунктир-ориентир: y ~ mu^power, проведён через последнюю точку (чуть смещён)."""
    mus, ys = np.array(mus, float), np.array(ys, float)
    y_ref = ys[-1] * (mus / mus[-1]) ** power * 2.2
    ax.plot(mus, y_ref, "--", color="gray", lw=1, label=label)


def make_plots(rows, hists, devs, tag, crit="gap", eps_gap=1e-4, m=3, n=3):
    mus = [r["mu"] for r in rows]
    C, R, Q = "#1f77b4", "#d62728", "#ff7f0e"

    # 1. обусловленность
    cond = [r["cond_H"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(mus, cond, "o-", color=C, label="cond(H) в x*_μ")
    _ref(ax, mus, cond, 1, "ориентир: ∝ μ (наклон 1)")
    for x, y in zip(mus, cond):
        ax.annotate(f"{y:.3g}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    _setup(ax, "Обусловленность растёт пропорционально μ", ylabel="cond(H) = λ_max / λ_min")
    ax.legend(); _save(fig, "cond_vs_mu", tag)

    # 2. точность ограничений и качество
    res = [r["residual"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(mus, res, "o-", color=R, label="невязка ‖Ax − r‖")
    ax.plot(mus, [abs(r["f_minus_fstar"]) for r in rows], "s-", color=Q, label="|f − f*|")
    _ref(ax, mus, res, -1, "ориентир: ∝ 1/μ (наклон −1)")
    for x, y in zip(mus, res):
        ax.annotate(f"{y:.2g}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color=R)
    _setup(ax, "Ограничения выполняются тем точнее, чем больше μ")
    ax.legend(); _save(fig, "quality_vs_mu", tag)

    # 3. итерации GD (не сошедшиеся точки показываем у потолка max_iter)
    key = f"gd_iters_{crit}"; ok = f"gd_converged_{crit}"
    crit_txt = (f"F_μ − F_μ* ≤ {eps_gap:g}" if crit == "gap" else "‖∇F‖ ≤ 1e-6·‖∇F(x₀)‖")
    fig, ax = plt.subplots(figsize=(6, 4))
    good = [(r["mu"], r[key]) for r in rows if r[ok]]
    bad = [(r["mu"], _num(r[key])) for r in rows if not r[ok]]
    ax.plot(*zip(*good), "o-", color=C, label="GD, шаг 1/L")
    if bad:
        ax.plot(*zip(*bad), "x", color=R, ms=10, mew=2, label="не сошёлся за лимит итераций")
        for x, y in bad:
            ax.annotate(f">{y}", (x, y), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8, color=R)
    for x, y in good:
        ax.annotate(f"{y}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    _ref(ax, [g[0] for g in good], [g[1] for g in good], 1, "ориентир: ∝ μ")
    _setup(ax, f"Число итераций GD ({crit_txt})", ylabel="итераций")
    ax.legend(loc="lower right", fontsize=8); _save(fig, "iters_vs_mu", tag)

    # 4. кривые сходимости
    fig, ax = plt.subplots(figsize=(7, 4))
    for mu, d in hists.items():
        ax.semilogy(np.maximum(d, 1e-16), label=f"μ = {mu:g}")
    ax.axhline(eps_gap, ls="--", color="gray", lw=1)
    ax.text(0.99, eps_gap * 1.5, f"порог ε = {eps_gap:g}", transform=ax.get_yaxis_transform(), ha="right", fontsize=8, color="gray")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("итерация GD"); ax.set_ylabel("F_μ(x_k) − F_μ*")
    ax.grid(True, which="both", alpha=.25); ax.legend(); ax.set_title("Сходимость GD: чем больше μ, тем медленнее")
    _save(fig, "convergence", tag)

    # 5. компромисс: точность против стоимости вычислений
    fig, ax = plt.subplots(figsize=(6, 4.3))
    xs = [_num(r[key]) for r in rows]; ys = res
    ax.plot(xs, ys, "-", color="gray", lw=1, zorder=1)
    for r, x, y in zip(rows, xs, ys):
        done = r[ok]
        ax.scatter([x], [y], s=70, zorder=2, color=C if done else "white", edgecolor=C if done else R, linewidth=2)
        ax.annotate(f"μ={r['mu']:g}" + ("" if done else "\n(не сошёлся)"), (x, y),
                    textcoords="offset points", xytext=(8, 6), fontsize=8)
    _setup(ax, "Компромисс: точнее ограничения — больше итераций",
           xlabel="итераций GD (стоимость вычислений)", ylabel="невязка ‖Ax − r‖ (ошибка ограничений)")
    ax.margins(x=0.15, y=0.15)
    _save(fig, "tradeoff", tag)

    # 6. что такое невязка физически: недовоз/перевоз по каждому складу и магазину
    pick = [mu for mu in mus if mu in (1, 10, 100, 1000)] or mus[:4]
    cols = 2; rws = int(np.ceil(len(pick) / cols))
    fig, axs = plt.subplots(rws, cols, figsize=(8, 3.2 * rws), squeeze=False)
    names = [f"Склад {i+1}" for i in range(m)] + [f"Магазин {j+1}" for j in range(n)]
    if m + n > 8:
        names = [f"С{i+1}" for i in range(m)] + [f"М{j+1}" for j in range(n)]
    for ax, mu in zip(axs.ravel(), pick):
        d = devs[mu]
        ax.bar(range(len(d)), d, color=[Q if v < 0 else C for v in d])
        ax.axhline(0, color="black", lw=.8); ax.axvline(m - .5, color="gray", ls=":")
        ax.set_xticks(range(len(d))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_title(f"μ = {mu:g}:  ‖Ax − r‖ = {np.linalg.norm(d):.3g}", fontsize=10)
        ax.set_ylabel("(Ax − r)ᵢ, ед. груза"); ax.grid(axis="y", alpha=.25)
    for ax in axs.ravel()[len(pick):]:
        ax.axis("off")
    fig.suptitle("Невязка по ограничениям: оранж. — недовоз/недопоставка, синий — перевоз/излишек\n(масштаб оси Y в каждой панели свой: с ростом μ столбцы уменьшаются в разы)", fontsize=10)
    fig.tight_layout(); _save(fig, "imbalance", tag)


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
    f_star, info, rows, hists, devs = run(a.m, a.n, a.seed, mus, a.tol, a.eps_gap, a.max_iter)
    print(f"\nf* (scipy) = {f_star:.8f}; проверки: {info}")

    tag = f"m{a.m}n{a.n}s{a.seed}"
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/metrics_table_{tag}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    make_plots(rows, hists, devs, tag, a.criterion, a.eps_gap, a.m, a.n)
    print(f"Таблица и графики сохранены в {OUT}/")
