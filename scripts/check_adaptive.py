"""Быстрая проверка AdaptiveStep: все стратегии на квадратичной задаче.

Запуск из корня репозитория:
    python -m scripts.check_adaptive
"""
import numpy as np

from oracles import QuadraticOracle
from methods import AdaptiveStep

TOL = 1e-4

oracle = QuadraticOracle(dim=50, kappa=1000, seed=0)
x0 = np.zeros(oracle.dim)
L = oracle.info.L

for strategy, eta0 in [("backtrack", 10.0), ("bb1", 1.0 / L), ("bb2", 1.0 / L), ("polyak", 1.0 / L)]:
    method = AdaptiveStep(oracle, strategy=strategy, eta0=eta0, max_iter=20000, tol=TOL)
    hist = method.run(x0)
    iters = len(hist.f) - 1
    status = "OK" if hist.grad_norm[-1] <= TOL else "НЕ СОШЁЛСЯ"
    print(f"{strategy:10s} итераций: {iters:6d}   ||grad|| = {hist.grad_norm[-1]:.1e}   {status}")