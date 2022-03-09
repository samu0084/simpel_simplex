import random
import time

from scipy.optimize import linprog
import numpy as np
from lpsolve import simple_simplex


def experiment_execution_time(seed_for_random, random_lp_choice, cmp_function, iterations, dtype, pivotrule_function):
    random.seed(seed_for_random)
    duration_sum_linprog = 0
    duration_sum_lp_solve = 0
    for i in range(iterations):
        n = random.randint(1, 50)
        m = random.randint(1, 50)
        duration_linprog, duration_lp_solve = compare_to_linprog(random_lp_choice, cmp_function, n, m, dtype,
                                                                 lambda d, eps: pivotrule_function(d, eps))
        duration_sum_linprog += duration_linprog
        duration_sum_lp_solve += duration_lp_solve
    average_linprog = duration_sum_linprog / 50
    average_lp_solve = duration_sum_lp_solve / 50
    print(f"{pivotrule_function}, {dtype}: Average linprog: {average_linprog}")
    print(f"{pivotrule_function}, {dtype}: Average lp_solve: {average_lp_solve}")


def compare_to_linprog(random_lp_choice, cmp_function, n, m, dtype, pivotrule=None):
    if random_lp_choice == "non-negative b-values":
        c, a, b = random_lp_only_none_negative_b_values(n, m)
    else:
        c, a, b = random_lp_including_negative_b_values(n, m)
    start_linprog = time.time()
    linprog(-c, a, b, method="simplex")
    end_linprog = time.time()
    duration_linprog = end_linprog - start_linprog

    start_lp_solve = time.time()
    cmp_function(c, a, b, dtype=dtype, pivotrule=pivotrule)
    end_lp_solve = time.time()
    duration_lp_solve = end_lp_solve - start_lp_solve

    return duration_linprog, duration_lp_solve


def random_lp_only_none_negative_b_values(n, m, sigma=10):
    return np.round(sigma * np.random.randn(n)), np.round(sigma * np.random.randn(m, n)), np.round(
        sigma * np.abs(np.random.randn(m)))


def random_lp_including_negative_b_values(n, m, sigma=10):
    return np.round(sigma * np.random.randn(n)), np.round(sigma * np.random.randn(m, n)), np.round(
        sigma * np.random.randn(m))

