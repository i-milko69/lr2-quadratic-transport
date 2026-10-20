"""Сравнение: LP-предел, Франк–Вульф и GD со штрафом на одной задаче.

Запуск:  python -m scripts.compare_linearization [--seed 42] [--m 3] [--n 3]

Все методы оцениваются на ИСХОДНОЙ стоимости sum(a x + b x^2 / 2) (оракул
с mu = 0) и по нарушению ограничений, чтобы сравнение было честным: у GD
со штрафом f включает штрафные слагаемые, и его собственные значения f
несопоставимы с остальными.
"""
import argparse
import numpy as np

from oracles.transport import TransportOracle
from methods.gd import GD
from methods.frank_wolfe import FrankWolfe, lp_plan, _constraint_matrix


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=3)
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gd-max-iter", type=int, default=100000)
    args = p.parse_args()

    true = TransportOracle(args.m, args.n, args.seed, mu=0.0)
    A = _constraint_matrix(args.m, args.n)
    rhs = np.concatenate([true.s, true.d])
    x0 = np.zeros(true.dim)

    # эталон: FW с очень малым зазором (зазор >= f - f*)
    ref = FrankWolfe(true, max_iter=100000, tol=1e-10).run(x0)
    f_star = ref.f[-1]

    rows = []

    def add(name, iters, x, note=""):
        rows.append((name, iters, true.f(x), true.f(x) - f_star,
                     float(np.abs(A @ x - rhs).max()), float(x.min()), note))

    xl = lp_plan(true)
    add("LP-предел (b->0)", "-", xl, "один LP")

    for rule in ("exact", "classic"):
        h = FrankWolfe(true, step_rule=rule, max_iter=20000, tol=1e-6).run(x0)
        add(f"FW {rule}", len(h.f) - 1, h.x[-1],
            f"gap={h.grad_norm[-1]:.1e}, LP на итерацию")

    for mu in (10.0, 100.0, 1000.0):
        o = TransportOracle(args.m, args.n, args.seed, mu=mu)
        h = GD(o, max_iter=args.gd_max_iter, tol=1e-6).run(x0)
        conv = "сошёлся" if h.grad_norm[-1] <= 1e-6 else "НЕ сошёлся"
        add(f"GD штраф mu={mu:g}", len(h.f) - 1, h.x[-1], conv)

    print(f"f* (по FW, tol=1e-10) = {f_star:.8f}\n")
    hdr = f"{'метод':22}{'итер':>8}{'f (исходная)':>16}{'f - f*':>12}" \
          f"{'нарушение огр.':>16}{'min x':>9}  примечание"
    print(hdr)
    print("-" * len(hdr))
    for name, it, f_, gap, viol, mn, note in rows:
        print(f"{name:22}{str(it):>8}{f_:16.6f}{gap:12.2e}{viol:16.2e}{mn:9.2f}  {note}")


if __name__ == "__main__":
    main()
