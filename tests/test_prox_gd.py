import numpy as np

from methods import ProxGD
from oracles.transport import TransportOracle


def test_prox_gd_returns_nonnegative_shipments():
    oracle = TransportOracle(m=3, n=3, seed=42, mu=100.0)
    method = ProxGD(oracle, max_iter=100, tol=1e-7)

    history = method.run(np.zeros(oracle.dim))
    x = history.x[-1]

    assert len(history.f) > 0
    assert np.all(x >= 0)
    assert np.isfinite(history.f[-1])
    assert np.isfinite(history.grad_norm[-1])


if __name__ == "__main__":
    test_prox_gd_returns_nonnegative_shipments()
    print("Все проверки ProxGD пройдены.")
