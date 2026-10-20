import numpy as np

def generate_data(m=3, n=3, seed=42):
    """
    Генерирует синтетические данные для транспортной задачи.
    m - количество складов, n - количество магазинов.
    """
    rng = np.random.default_rng(seed)
    
    # 1. Генерируем тарифы (a) и коэффициенты пробок (b)
    a = rng.uniform(1.0, 10.0, size=(m, n))
    b = rng.uniform(0.1, 2.0, size=(m, n))  # b > 0 для сильной выпуклости
    
    # 2. Генерируем спрос и предложение (чтобы сумма совпадала!)
    supply_raw = rng.uniform(10, 100, size=m)
    demand_raw = rng.uniform(10, 100, size=n)
    
    # Балансировка: приводим суммы к одному значению (например, к 100)
    total = 100.0
    s = supply_raw * (total / supply_raw.sum())
    d = demand_raw * (total / demand_raw.sum())
    
    return a, b, s, d

def f(x, a, b, s, d, mu=100.0):
    """
    Целевая функция со штрафами.
    x - матрица (m x n) перевозок
    """
    # 1. Исходная квадратичная стоимость
    f_orig = np.sum(a * x + 0.5 * b * x**2)
    
    # 2. Штраф за небаланс по складам
    row_sums = x.sum(axis=1)
    penalty_rows = np.sum((row_sums - s)**2)
    
    # 3. Штраф за небаланс по магазинам
    col_sums = x.sum(axis=0)
    penalty_cols = np.sum((col_sums - d)**2)
    
    # 4. Штраф за отрицательные перевозки (x >= 0)
    penalty_neg = np.sum(np.maximum(0, -x)**2)
    
    # Суммируем всё с коэффициентом штрафа mu
    F_mu = f_orig + 0.5 * mu * (penalty_rows + penalty_cols + penalty_neg)
    return F_mu

def grad(x, a, b, s, d, mu=100.0):
    """
    Градиент целевой функции со штрафами.
    """
    m, n = x.shape
    
    # 1. Градиент исходной стоимости: a + b*x
    grad_orig = a + b * x
    
    # 2. Градиент штрафа по складам
    row_diff = x.sum(axis=1) - s
    grad_rows = np.tile(row_diff.reshape(-1, 1), (1, n))
    
    # 3. Градиент штрафа по магазинам
    col_diff = x.sum(axis=0) - d
    grad_cols = np.tile(col_diff.reshape(1, -1), (m, 1))
    
    # 4. Градиент штрафа за отрицательность
    grad_neg = -2 * np.maximum(0, -x)
    
    # Собираем итоговый градиент
    g = grad_orig + mu * (grad_rows + grad_cols + grad_neg)
    return g
