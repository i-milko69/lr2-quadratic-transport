"""Сравнение для защиты: ProxGD (ёмкости склада) против эталона и FW.

Запуск:  python -m scripts.compare_prox [--slack 0.2] [--out results/prox]

Все методы оцениваются на исходной стоимости sum(a x + b x^2/2) и по
нарушению ограничений. Эталон f* — SLSQP, Франк-Вульф даёт допустимую точку
и гарантированную оценку зазора.
"""
import argparse
import time
from pathlib import Path

import numpy as np

from methods import ProxGD, FrankWolfe
from oracles.transport import TransportOracle
from utils.reference import reference_solution


def violations(o, x):
    xm = np.asarray(x).reshape(o.m, o.n)
    return (float(max(0.0, (xm.sum(axis=1) - o.capacity).max())),
            float(np.abs(xm.sum(axis=0) - o.d).max()),
            float(max(0.0, -xm.min())))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=3)
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--slack", type=float, default=0.2)
    p.add_argument("--out", default="results/prox")
    args = p.parse_args()

    o_true = TransportOracle(args.m, args.n, args.seed, mu=0.0,
                             capacity_slack=args.slack)
    x_ref, f_star = reference_solution(o_true)
    x0 = np.zeros(o_true.dim)

    rows, hists = [], {}
    for mu in (10.0, 100.0, 1000.0):
        o = TransportOracle(args.m, args.n, args.seed, mu=mu,
                            capacity_slack=args.slack)
        t = time.perf_counter()
        h = ProxGD(o, max_iter=400_000, tol=1e-6).run(x0)
        dt = time.perf_counter() - t
        hists[f"ProxGD mu={mu:g}"] = h
        x = h.x[-1]
        rows.append((f"ProxGD mu={mu:g}", len(h.f) - 1, o_true.true_cost(x),
                     *violations(o_true, x), dt))

    t = time.perf_counter()
    h = FrankWolfe(o_true, max_iter=3000, tol=1e-3).run(x0)
    rows.append(("Frank-Wolfe", len(h.f) - 1, o_true.true_cost(h.x[-1]),
                 *violations(o_true, h.x[-1]), time.perf_counter() - t))

    xm = x_ref.reshape(o_true.m, o_true.n)
    print(f"f* (SLSQP) = {f_star:.6f}")
    print("ёмкости C:       ", np.round(o_true.capacity, 3))
    print("загрузка склада: ", np.round(xm.sum(axis=1), 3), "(эталон)\n")
    hdr = (f"{'метод':18}{'итер':>8}{'f':>12}{'f-f*':>10}"
           f"{'нарушение C':>13}{'наруш. спроса':>15}{'min x<0':>9}{'время,с':>9}")
    print(hdr); print("-" * len(hdr))
    for name, it, f_, vc, vd, vn, dt in rows:
        print(f"{name:18}{it:8d}{f_:12.4f}{f_ - f_star:10.2e}"
              f"{vc:13.1e}{vd:15.1e}{vn:9.1e}{dt:9.2f}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for name, h in hists.items():
            ax[0].plot(h.f, label=name)
            ax[1].semilogy(h.grad_norm, label=name)
        ax[0].set_xscale("log"); ax[0].set_xlabel("итерация")
        ax[0].set_ylabel("F_mu(x) (со штрафом)"); ax[0].legend()
        ax[1].set_xscale("log"); ax[1].set_xlabel("итерация")
        ax[1].set_ylabel("||G(x)|| (прокс-стационарность)")
        fig.tight_layout(); fig.savefig(out / "prox_convergence.png", dpi=150)
        print(f"\nГрафик: {out / 'prox_convergence.png'}")
    except ImportError:
        print("matplotlib не установлен, график пропущен")


if __name__ == "__main__":
    main()
