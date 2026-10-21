
import numpy as np

from oracles.transport import TransportOracle
from methods.gd import GD


def constraint_residual(oracle, x):
    X = np.asarray(x).reshape(oracle.m, oracle.n)

    row_error = np.max(np.abs(X.sum(axis=1) - oracle.s))
    col_error = np.max(np.abs(X.sum(axis=0) - oracle.d))

    return max(row_error, col_error)


def main():
    m, n, seed = 3, 3, 42
    max_iter = 1000
    tol = 1e-8

    # Одна и та же транспортная задача для всех запусков.
    initial_oracle = TransportOracle(
        m=m, n=n, seed=seed, mu=1000
    )
    x0 = np.zeros(initial_oracle.dim)

    # Базовый метод: сразу используем большой штраф.
    baseline = GD(
        initial_oracle,
        max_iter=max_iter,
        tol=tol,
    ).run(x0)

    x_baseline = baseline.x[-1].copy()

    # Метод продолжения: постепенно увеличиваем штраф.
    x = x0.copy()
    total_iterations = 0

    for mu in (10, 100, 1000):
        oracle = TransportOracle(
            m=m, n=n, seed=seed, mu=mu
        )

        result = GD(
            oracle,
            max_iter=max_iter,
            tol=tol,
        ).run(x)

        x = result.x[-1].copy()
        total_iterations += len(result.f) - 1

        print(
            f"mu={mu}: "
            f"итераций={len(result.f) - 1}, "
            f"F_mu={result.f[-1]:.6f}"
        )

    # Исходная стоимость без штрафов.
    cost_oracle = TransportOracle(
        m=m, n=n, seed=seed, mu=0
    )

    # Функция и ограничения при конечном штрафе mu=1000.
    final_oracle = TransportOracle(
        m=m, n=n, seed=seed, mu=1000
    )

    print("\nИТОГОВОЕ СРАВНЕНИЕ")
    print("-" * 65)

    for name, solution, iterations in [
        ("Обычный GD", x_baseline, len(baseline.f) - 1),
        ("Continuation", x, total_iterations),
    ]:
        print(
            f"\n{name}\n"
            f"Итераций: {iterations}\n"
            f"Исходная стоимость: {cost_oracle.f(solution):.6f}\n"
            f"Штрафная функция: {final_oracle.f(solution):.6f}\n"
            f"Невязка ограничений: "
            f"{constraint_residual(cost_oracle, solution):.6e}\n"
            f"Норма градиента: "
            f"{np.linalg.norm(final_oracle.grad(solution)):.6f}"
        )


if __name__ == "__main__":
    main()
