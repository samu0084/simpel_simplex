from fractions import Fraction

import numpy as np

from experiments import experiment_execution_time
from lpsolve import simple_simplex, lp_solve
from pivotrules import bland, largest_coefficient, largest_increase


def main():
    seed_for_random = 8
    iterations = 50
    random_lp_choice = "non-negative b-values"

    experiment_execution_time(seed_for_random, random_lp_choice, simple_simplex, iterations, np.float64, bland)
    experiment_execution_time(seed_for_random, random_lp_choice, simple_simplex, iterations, Fraction, bland)
    experiment_execution_time(seed_for_random, random_lp_choice, simple_simplex, iterations, int, bland)
    experiment_execution_time(seed_for_random, random_lp_choice, simple_simplex, iterations, np.float64, largest_coefficient)
    experiment_execution_time(seed_for_random, random_lp_choice, simple_simplex, iterations, Fraction, largest_coefficient)
    experiment_execution_time(seed_for_random, random_lp_choice, simple_simplex, iterations, int, largest_coefficient)
    experiment_execution_time(seed_for_random, random_lp_choice, simple_simplex, iterations, np.float64, largest_increase)
    experiment_execution_time(seed_for_random, random_lp_choice, simple_simplex, iterations, Fraction, largest_increase)
    experiment_execution_time(seed_for_random, random_lp_choice, simple_simplex, iterations, int, largest_increase)

    random_lp_choice = "including negative b-values"

    experiment_execution_time(seed_for_random, random_lp_choice, lp_solve, iterations, np.float64, bland)
    experiment_execution_time(seed_for_random, random_lp_choice, lp_solve, iterations, Fraction, bland)
    experiment_execution_time(seed_for_random, random_lp_choice, lp_solve, iterations, int, bland)
    experiment_execution_time(seed_for_random, random_lp_choice, lp_solve, iterations, np.float64, largest_coefficient)
    experiment_execution_time(seed_for_random, random_lp_choice, lp_solve, iterations, Fraction, largest_coefficient)
    experiment_execution_time(seed_for_random, random_lp_choice, lp_solve, iterations, int, largest_coefficient)
    experiment_execution_time(seed_for_random, random_lp_choice, lp_solve, iterations, np.float64, largest_increase)
    experiment_execution_time(seed_for_random, random_lp_choice, lp_solve, iterations, Fraction, largest_increase)
    experiment_execution_time(seed_for_random, random_lp_choice, lp_solve, iterations, int, largest_increase)

    print("Done maaaain!")


if __name__ == "__main__":
    main()
