from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

from oracles import TransportOracle
from methods import GD, AdaptiveStep

MUS = [1, 10, 100, 1000]
REL_TOL = 1e-6
MAX_ITER = 100_000
OUT_DIR = Path("results/plots")

COLORS = {"GD, шаг 1/L": "#2a78d6", "Армихо": "#eb6834", "BB1": "#1baf7a"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def make_methods(oracle, tol):
    L = oracle.info.L
    return {
        "GD, шаг 1/L": GD(oracle, max_iter=MAX_ITER, tol=tol),
        "Армихо": AdaptiveStep(oracle, strategy="backtrack", eta0=1.0,
                               max_iter=MAX_ITER, tol=tol),
        "BB1": AdaptiveStep(oracle, strategy="bb1", eta0=1.0 / L,
                            max_iter=MAX_ITER, tol=tol),
        "BB2": AdaptiveStep(oracle, strategy="bb2", eta0=1.0 / L,
                            max_iter=MAX_ITER, tol=tol),
    }


def run_all():
    table, histories = [], {}
    for mu in MUS:
        oracle = TransportOracle(m=3, n=3, seed=42, mu=mu)
        x0 = np.zeros(oracle.dim)
        g0 = np.linalg.norm(oracle.grad(x0))
        tol = REL_TOL * g0

        eig = np.linalg.eigvalsh(oracle.hess(x0))
        row = {"mu": mu, "kappa": eig[-1] / eig[0]}

        for name, method in make_methods(oracle, tol).items():
            hist = method.run(x0)
            converged = hist.grad_norm[-1] <= tol
            row[name] = len(hist.f) - 1 if converged else None
            histories[(mu, name)] = np.array(hist.grad_norm) / g0
        table.append(row)
        print(f"mu = {mu}: готово")
    return table, histories


def print_table(table):
    names = ["GD, шаг 1/L", "Армихо", "BB1", "BB2"]
    print("\n| μ | κ | " + " | ".join(names) + " |")
    print("|" + "---|" * (len(names) + 2))
    for row in table:
        cells = [f"{row[n]:,}".replace(",", " ") if row[n] else "не сошёлся"
                 for n in names]
        print(f"| {row['mu']} | {row['kappa']:,.0f} | ".replace(",", " ")
              + " | ".join(cells) + " |")


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=13, color=INK, pad=12)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK_2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)


def plot_iterations(table):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, color in COLORS.items():
        ys = [row[name] for row in table]
        ax.plot(MUS, ys, color=color, linewidth=2, marker="o", markersize=7, label=name)
        label = f"{name}: " + f"{ys[-1]:,}".replace(",", " ")
        ax.annotate(label, (MUS[-1], ys[-1]),
                    xytext=(10, 0), textcoords="offset points", va="center",
                    fontsize=10, color=INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.7, 1.4e4)
    ax.set_xticks(MUS, [str(m) for m in MUS])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("коэффициент штрафа μ", color=INK_2)
    ax.set_ylabel("итераций до сходимости", color=INK_2)
    style(ax, "Чем больше штраф, тем дольше GD и Армихо. BB почти не замечает μ")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    path = OUT_DIR / "adaptive_iters_vs_mu.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_convergence(histories, table, mu=100, first=300):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, color in COLORS.items():
        h = histories[(mu, name)][: first + 1]
        ax.semilogy(np.arange(len(h)), h, color=color, linewidth=2, label=name)
    ax.axhline(REL_TOL, color=INK_2, linewidth=1, linestyle="--")
    ax.annotate("цель: градиент меньше в 10⁶ раз", (first, REL_TOL), xytext=(0, 6),
                textcoords="offset points", ha="right", fontsize=10, color=INK_2)
    ax.set_xlim(0, first)
    ax.set_xlabel("итерация", color=INK_2)
    ax.set_ylabel("длина градиента / в старте", color=INK_2)
    bb_iters = next(row["BB1"] for row in table if row["mu"] == mu)
    style(ax, f"μ = {mu}: BB у цели за {bb_iters} итераций, GD и Армихо всё ещё ползут")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    path = OUT_DIR / f"adaptive_convergence_mu{mu}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table, histories = run_all()
    print_table(table)
    p1 = plot_iterations(table)
    p2 = plot_convergence(histories, table)
    print(f"\nГрафики сохранены:\n  {p1}\n  {p2}")


if __name__ == "__main__":
    main()