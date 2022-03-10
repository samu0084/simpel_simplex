import math
from fractions import Fraction
from unittest import TestCase

import numpy as np
from numpy import random
from scipy.optimize import linprog

import dictionary
from dictionary import Dictionary
from experiments import random_lp_including_negative_b_values, random_lp_only_none_negative_b_values
from lpresult import LPResult
from lpsolve import lp_solve
from pivotrules import leaving_variable
from pivotrules import bland
from pivotrules import largest_coefficient
from pivotrules import largest_increase


class Test(TestCase):
    def test_leaving_variable(self):
        verbose = True
        c = np.array([7, 4])
        a = np.array([[2, 1],
                      [1, 1],
                      [1, 0]])
        b = np.array([20, 18, 8])
        d = Dictionary(c, a, b)
        if verbose:
            print(d)
            print()
        # Test leaving_variable for entering 0
        entering = 0
        expected = 2
        leaving, ratio = leaving_variable(d, 0, entering, verbose)
        self.assertEqual(expected, leaving)
        # Test leaving_variable for entering 1
        entering = 1
        expected = 1
        print(d)
        leaving, ratio = leaving_variable(d, 0, entering, verbose)
        self.assertEqual(expected, leaving)


    def test_leaving_variable1(self):
        verbose = False
        # Make dictionary
        c = np.array([5, 4, 3])
        a = np.array([[2, 3, 1],
                      [4, 1, 2],
                      [3, 4, 2]])
        b = np.array([5, 11, 8])
        d = Dictionary(c, a, b)
        if verbose:
            print(d)
            print()
        # Test leaving_variable for entering 0
        entering = 0
        expected = 0
        leaving, ratio = leaving_variable(d, 0, entering, verbose)
        self.assertEqual(expected, leaving)
        # Test leaving_variable for entering 1
        entering = 1
        expected = 0
        leaving, ratio = leaving_variable(d, 0, entering, verbose)
        self.assertEqual(expected, leaving)
        # Test leaving_variable for entering 2
        entering = 2
        expected = 2
        leaving, ratio = leaving_variable(d, 0, entering, verbose)
        self.assertEqual(expected, leaving)

    def test_bland(self):
        verbose = False
        # Make dictionary
        c = np.array([7, 4])
        a = np.array([[2, 1],
                      [1, 1],
                      [1, 0]])
        b = np.array([20, 18, 8])
        d = Dictionary(c, a, b)
        if verbose:
            print(d)
            print()
        # Test blends rule
        entering_expected, leaving_expected = (0, 2)
        entering, leaving = bland(d, 0, verbose)
        self.assertEqual((entering_expected, leaving_expected), (entering, leaving))

    def test_bland2(self):
        verbose = False
        # Make dictionary
        c = np.array([5, 4, 3])
        a = np.array([[2, 3, 1],
                      [4, 1, 2],
                      [3, 4, 2]])
        b = np.array([5, 11, 8])
        d = Dictionary(c, a, b)
        if verbose:
            print(d)
            print()
        # Test blends rule
        entering_expected, leaving_expected = (0, 0)
        entering, leaving = bland(d, 0, verbose)
        self.assertEqual((entering_expected, leaving_expected), (entering, leaving))

    def test_largest_coefficient(self):
        verbose = False
        # Make dictionary
        c = np.array([7, 4])
        a = np.array([[2, 1],
                      [1, 1],
                      [1, 0]])
        b = np.array([20, 18, 8])
        d = Dictionary(c, a, b)
        if verbose:
            print(d)
            print()
        # Test largest_coefficient rule
        entering_expected, leaving_expected = (0, 2)
        entering, leaving = largest_coefficient(d, 0, verbose)
        self.assertEqual((entering_expected, leaving_expected), (entering, leaving))

    def test_largest_coefficient2(self):
        verbose = False
        # Make dictionary
        c = np.array([5, 4, 3])
        a = np.array([[2, 3, 1],
                      [4, 1, 2],
                      [3, 4, 2]])
        b = np.array([5, 11, 8])
        d = Dictionary(c, a, b)
        if verbose:
            print(d)
            print()
        # Test largest_coefficient rule
        entering_expected, leaving_expected = (0, 0)
        entering, leaving = largest_coefficient(d, 0, verbose)
        self.assertEqual((entering_expected, leaving_expected), (entering, leaving))

    def test_largest_coefficient3(self):
        verbose = False
        # Make dictionary
        c = np.array([5, 6, 3])
        a = np.array([[2, 3, 1],
                      [4, 7, 2],
                      [3, 4, 2]])
        b = np.array([5, 11, 8])
        d = Dictionary(c, a, b)
        if verbose:
            print(d)
            print()
        # Test largest_coefficient rule
        entering_expected, leaving_expected = (1, 1)
        entering, leaving = largest_coefficient(d, 0, verbose)
        self.assertEqual((entering_expected, leaving_expected), (entering, leaving))

    def test_largest_increase4(self):
        verbose = False
        c = np.array([2, 3])
        a = np.array([[1, 2],
                      [1, -1]])
        b = np.array([2, 3])
        d = Dictionary(c, a, b)
        if verbose:
            print(d)
            print()
        entering_expected, leaving_expected = (0, 0)
        entering, leaving = largest_increase(d, 0, verbose)
        self.assertEqual((entering_expected, leaving_expected), (entering, leaving))

    def test_largest_increase(self):
        verbose = False
        c = np.array([7, 4])
        a = np.array([[2, 1],
                      [1, 1],
                      [1, 0]])
        b = np.array([20, 18, 8])
        d = Dictionary(c, a, b)
        if verbose:
            print(d)
            print()
        # Test largest_coefficient rule
        entering_expected, leaving_expected = (1, 1)
        entering, leaving = largest_increase(d, 0, verbose)
        self.assertEqual((entering_expected, leaving_expected), (entering, leaving))

    def test_largest_increase2(self):
        verbose = False
        # Make dictionary
        c = np.array([5, 4, 3])
        a = np.array([[2, 3, 1],
                      [4, 1, 2],
                      [3, 4, 2]])
        b = np.array([5, 11, 8])
        d = Dictionary(c, a, b)
        if verbose:
            print(d)
            print()
        # Test largest_coefficient rule
        entering_expected, leaving_expected = (0, 0)
        entering, leaving = largest_increase(d, 0, verbose)
        self.assertEqual((entering_expected, leaving_expected), (entering, leaving))

    def test_largest_increase3(self):
        verbose = False
        # Make dictionary
        c = np.array([5, 6, 3])
        a = np.array([[2, 3, 1],
                      [4, 7, 2],
                      [3, 4, 2]])
        b = np.array([5, 11, 8])
        d = Dictionary(c, a, b)
        if verbose:
            print(d)
            print()
        # Test largest_coefficient rule
        entering_expected, leaving_expected = (0, 0)
        entering, leaving = largest_increase(d, 0, verbose)
        self.assertEqual((entering_expected, leaving_expected), (entering, leaving))

    def test_unbounded_largest_coefficient_one(self):
        verbose = False
        c = np.array([1])
        a = np.array([[-13]])
        b = np.array([2])
        d = dictionary.Dictionary(c, a, b, np.float64)

        entering_expected, leaving_expected = (0, None)
        entering, leaving = largest_coefficient(d, 0, verbose)
        self.assertEqual((entering_expected, leaving_expected), (entering, leaving))

        res, d = lp_solve(c, a, b, np.float64, pivotrule=lambda d, eps: largest_coefficient(d, eps, verbose))
        self.assertEqual(LPResult.UNBOUNDED, res)

    def test_unbounded_bland_one(self):
        verbose = False
        c = np.array([1])
        a = np.array([[-13]])
        b = np.array([2])
        d = dictionary.Dictionary(c, a, b, np.float64)
        print(d)
        entering_expected, leaving_expected = (0, None)
        entering, leaving = bland(d, 0, verbose)
        self.assertEqual((entering_expected, leaving_expected), (entering, leaving))

        res, d = lp_solve(c, a, b, np.float64, pivotrule=lambda d, eps: bland(d, eps, verbose))
        self.assertEqual(LPResult.UNBOUNDED, res)

    def test_unbounded_largest_coefficient_two(self):
        verbose = False
        c = np.array([1])
        a = np.array([[-13]])
        b = np.array([2])
        d = dictionary.Dictionary(c, a, b, np.float64)
        print(d)
        entering_expected, leaving_expected = (0, None)
        entering, leaving = largest_coefficient(d, 0, verbose)
        self.assertEqual((entering_expected, leaving_expected), (entering, leaving))

        res, d = lp_solve(c, a, b, np.float64, pivotrule=lambda d, eps: largest_coefficient(d, eps, verbose))
        self.assertEqual(LPResult.UNBOUNDED, res)

    def test_unbounded_largest_increase_one(self):
        verbose = False
        c = np.array([1])
        a = np.array([[-13]])
        b = np.array([2])
        d = dictionary.Dictionary(c, a, b, np.float64)
        print(d)
        entering_expected, leaving_expected = (math.inf, None)
        entering, leaving = largest_increase(d, 0, verbose)
        self.assertEqual((entering_expected, leaving_expected), (entering, leaving))

        res, d = lp_solve(c, a, b, np.float64, pivotrule=lambda d, eps: largest_increase(d, eps, verbose))
        self.assertEqual(LPResult.UNBOUNDED, res)

    def test_largest_increase_two(self):
        eps = 0.0000001
        verbose = False
        c = np.array([5, 6, 4])
        a = -np.array([[17.0, -9.0, 7.0],
                       [13.0, 1.0, 10.0],
                       [2.0, 2.0, -1.0],
                       [-7.0, 7.0, -4.0],
                       [-11.0, -1.0, -15.0],
                       [6.0, -3.0, 12.0]])
        b = np.array([12, 13, 10, 0, 10, 0])
        d = dictionary.Dictionary(c, a, b, np.float64)
        print(d)
        res, d = lp_solve(c, a, b, np.float64, pivotrule=lambda d, eps: largest_increase(d, eps, verbose))
        res_linprog = linprog(-c, a, b, method="simplex")
        if res == LPResult.OPTIMAL and not d.value() - eps <= -res_linprog.fun <= d.value() + eps:
            print(res_linprog)
            print(d)
            print(d.value())
            self.assertTrue(d.value() - eps <= -res_linprog.fun <= d.value() + eps)
        if (res == LPResult.OPTIMAL) != res_linprog.success:
            print(f"res == LPResult.OPTIMAL: {res == LPResult.OPTIMAL}")
            print(f"res_linprog.success: {res_linprog.success}")
            print(dictionary.Dictionary(c, a, b, dtype=int))
            print(res_linprog)
            self.assertTrue(False)

    """z = 0.0 + 5.0 * x1 + 6.0 * x2 + 4.0 * x3
    x4 = 12.0 + 17.0 * x1 - 9.0 * x2 + 7.0 * x3
    x5 = 13.0 + 13.0 * x1 + 1.0 * x2 + 10.0 * x3
    x6 = 10.0 + 2.0 * x1 + 2.0 * x2 - 1.0 * x3
    x7 = 0.0 - 7.0 * x1 + 7.0 * x2 - 4.0 * x3
    x8 = 10.0 - 11.0 * x1 - 1.0 * x2 - 15.0 * x3
    x9 = 0.0 + 6.0 * x1 - 3.0 * x2 + 12.0 * x3"""

    def test_for_correct_result_with_repeated_use_of_largest_increase(self):
        random.seed(10)
        eps = 0.00000001
        for i in range(2000):
            n = random.randint(1, 10)
            m = random.randint(1, 10)
            c, a, b = random_lp_only_none_negative_b_values(n,
                                                            m)  # random_lp_only_none_negative_b_values(n, m)  #random_lp_including_negative_b_values(n, m)
            res_linprog = linprog(-c, a, b, method="simplex")
            res, d = lp_solve(c, a, b, dtype=np.float64, pivotrule=lambda d, eps: largest_increase(d, eps))
            if res == LPResult.OPTIMAL and not d.value() - eps <= -res_linprog.fun <= d.value() + eps:
                print(res_linprog)
                print(d)
                print(d.value())
                self.assertTrue(d.value() - eps <= -res_linprog.fun <= d.value() + eps)
            if (res == LPResult.OPTIMAL) != res_linprog.success:
                print(f"res == LPResult.OPTIMAL: {res == LPResult.OPTIMAL}")
                print(f"res_linprog.success: {res_linprog.success}")
                print(dictionary.Dictionary(c, a, b, dtype=int))
                print(res_linprog)
                self.assertTrue(False)

    def test_non_negative_b_bland(self):
        self.assertTrue(
            iterative_results_comparison(1, 20, True, lp_solve, np.float64, lambda d, eps: bland(d, eps),
                                         0.0000001))
        self.assertTrue(
            iterative_results_comparison(1, 20, True, lp_solve, Fraction, lambda d, eps: bland(d, eps),
                                         0.0000001))
        self.assertTrue(
            iterative_results_comparison(1, 20, True, lp_solve, int, lambda d, eps: bland(d, eps), 0.0000001))

    def test_non_negative_b_largest_coefficient(self):
        self.assertTrue(iterative_results_comparison(1, 20, True, lp_solve, np.float64,
                                                     lambda d, eps: largest_coefficient(d, eps), 0.0000001))
        self.assertTrue(iterative_results_comparison(1, 20, True, lp_solve, Fraction,
                                                     lambda d, eps: largest_coefficient(d, eps), 0.0000001))
        self.assertTrue(iterative_results_comparison(1, 20, True, lp_solve, int,
                                                     lambda d, eps: largest_coefficient(d, eps), 0.0000001))

    def test_non_negative_b_largest_increase(self):
        self.assertTrue(iterative_results_comparison(1, 20, True, lp_solve, np.float64,
                                                     lambda d, eps: largest_increase(d, eps), 0.0000001))
        self.assertTrue(iterative_results_comparison(1, 20, True, lp_solve, Fraction,
                                                     lambda d, eps: largest_increase(d, eps), 0.0000001))
        self.assertTrue(
            iterative_results_comparison(1, 20, True, lp_solve, int, lambda d, eps: largest_increase(d, eps),
                                         0.0000001))

    def test_allowing_negative_b_bland(self):
        self.assertTrue(
            iterative_results_comparison(1, 20, False, lp_solve, np.float64, lambda d, eps: bland(d, eps),
                                         0.0000001))
        self.assertTrue(
            iterative_results_comparison(1, 20, False, lp_solve, Fraction, lambda d, eps: bland(d, eps),
                                         0.0000001))
        self.assertTrue(
            iterative_results_comparison(1, 20, False, lp_solve, int, lambda d, eps: bland(d, eps),
                                         0.0000001))

    def test_allowing_negative_b_largest_increase(self):
        self.assertTrue(iterative_results_comparison(1, 20, False, lp_solve, np.float64,
                                                     lambda d, eps: largest_increase(d, eps), 0.0000001))
        self.assertTrue(iterative_results_comparison(1, 20, False, lp_solve, Fraction,
                                                     lambda d, eps: largest_increase(d, eps), 0.0000001))
        self.assertTrue(
            iterative_results_comparison(1, 20, False, lp_solve, int, lambda d, eps: largest_increase(d, eps),
                                         0.0000001))

    def test_allowing_negative_b_largest_coefficient(self):
        self.assertTrue(iterative_results_comparison(1, 20, False, lp_solve, np.float64,
                                                     lambda d, eps: largest_coefficient(d, eps), 0.0000001))
        self.assertTrue(iterative_results_comparison(1, 20, False, lp_solve, Fraction,
                                                     lambda d, eps: largest_coefficient(d, eps), 0.0000001))
        self.assertTrue(iterative_results_comparison(2, 20, False, lp_solve, dtype=int,
                                                     pivotrule=lambda d, eps: largest_coefficient(d, eps), eps=0.0000001))

def iterative_results_comparison(seed, iterations, only_none_negative_b_values, our_simplex, dtype, pivotrule=None,
                                 eps=0):
    random.seed(seed)
    for i in range(iterations):
        n = random.randint(1, 50)
        m = random.randint(1, 50)
        if not compare_results_to_linprog(only_none_negative_b_values, our_simplex, n, m, dtype, pivotrule, eps):
            print(f"none_negative_b failed at iteration {i}")
            return False
    print(f"none_negative_b succeeded all {iterations} iterations")
    return True


def compare_results_to_linprog(only_none_negative_b_values, our_simplex, n, m, dtype, pivotrule=None, eps=0):
    if only_none_negative_b_values:
        c, a, b = random_lp_only_none_negative_b_values(n, m)
    else:
        c, a, b = random_lp_including_negative_b_values(n, m)
    res_linprog = linprog(-c, a, b, method="simplex")
    res_our_simplex, d = our_simplex(c, a, b, dtype=dtype, pivotrule=pivotrule)
    if (res_our_simplex == LPResult.OPTIMAL) != res_linprog.success:
        print(f"Ours: {res_our_simplex}, linprog.success: {res_linprog.success}")
        print("Initial dictionary:")
        print(dictionary.Dictionary(c, a, b, dtype))
        if res_our_simplex == LPResult.OPTIMAL:
            print("Supposedly optimal dictionary")
            print(d)
        print(res_linprog)
        return False
    elif res_our_simplex == LPResult.OPTIMAL and not d.value() - eps <= -res_linprog.fun <= d.value() + eps:
        print(f"Our optimal dictionary:")
        print(d)
        print(f"Optimal value: {d.value()}")
        print(f"Linprog value: {-res_linprog.fun}")
        return False
    return True
