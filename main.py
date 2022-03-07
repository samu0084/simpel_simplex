from fractions import Fraction

import numpy as np

from experiments import experiment_simple_simplex
from pivotrules import bland, largest_coefficient, largest_increase


def main():
    iterations = 5
    experiment_simple_simplex(iterations, np.float64, bland)
    experiment_simple_simplex(iterations, Fraction, bland)
    experiment_simple_simplex(iterations, int, bland)
    experiment_simple_simplex(iterations, np.float64, largest_coefficient)
    experiment_simple_simplex(iterations, Fraction, largest_coefficient)
    experiment_simple_simplex(iterations, int, largest_coefficient)
    experiment_simple_simplex(iterations, np.float64, largest_increase)
    experiment_simple_simplex(iterations, Fraction, largest_increase)
    experiment_simple_simplex(iterations, int, largest_increase)
    # TODO: Test integer pivoting for the two-phase simplex method

    print("Done maaaain!")


if __name__ == "__main__":
    main()
