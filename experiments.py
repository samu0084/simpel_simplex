import random
import time

from scipy.optimize import linprog
import numpy as np
from lpsolve import simple_simplex


def experiment_simple_simplex(iterations, dtype, pivotrule_function):
    random.seed()
    duration_sum_linprog = 0
    duration_sum_lp_solve = 0
    for i in range(iterations):
        n = random.randint(1, 50)
        m = random.randint(1, 50)
        duration_linprog, duration_lp_solve = run_test_dict(n, m, dtype, lambda d, eps: pivotrule_function(d, eps))
        duration_sum_linprog += duration_linprog
        duration_sum_lp_solve += duration_lp_solve
    average_linprog = duration_sum_linprog / 50
    average_lp_solve = duration_sum_lp_solve / 50
    print(f"{pivotrule_function}, {dtype}: Average linprog: {average_linprog}")
    print(f"{pivotrule_function}, {dtype}: Average lp_solve: {average_lp_solve}")


def run_test_dict(n, m, dtype, pivotrule=None):
    c, a, b = random_lp(n, m)
    start_linprog = time.time()
    linprog(-c, a, b, method="simplex")
    end_linprog = time.time()
    duration_linprog = end_linprog - start_linprog

    start_lp_solve = time.time()
    simple_simplex(c, a, b, dtype=dtype, pivotrule=pivotrule)
    end_lp_solve = time.time()
    duration_lp_solve = end_lp_solve - start_lp_solve

    return duration_linprog, duration_lp_solve


def random_lp(n, m, sigma=10):
    return np.round(sigma * np.random.randn(n)), np.round(sigma * np.random.randn(m, n)), np.round(
        sigma * np.abs(np.random.randn(m)))


def random_lp_with_negative_b_values(n, m, sigma=10):
    return np.round(sigma * np.random.randn(n)), np.round(sigma * np.random.randn(m, n)), np.round(
        sigma * np.random.randn(m))

