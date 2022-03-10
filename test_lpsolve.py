from datetime import time
from fractions import Fraction
from math import copysign
import random
from unittest import TestCase

import numpy as np

import dictionary
import lpsolve
from dictionary import Dictionary
from experiments import compare_to_linprog, random_lp_only_none_negative_b_values, random_lp_including_negative_b_values
from lpresult import LPResult
from lpsolve import lp_solve, lp_solve_two_phase
from scipy.optimize import linprog as linprog_original, linprog

from pivotrules import bland


def example1():
    return np.array([5, 4, 3]), np.array([[2, 3, 1], [4, 1, 2], [3, 4, 2]]), np.array([5, 11, 8])


def example2():
    return np.array([-2, -1]), np.array([[-1, 1], [-1, -2], [0, 1]]), np.array([-1, -2, 1])


def integer_pivoting_example():
    return np.array([5, 2]), np.array([[3, 1], [2, 5]]), np.array([7, 5])


def exercise2_5():
    return np.array([1, 3]), np.array([[-1, -1], [-1, 1], [1, 2]]), np.array([-3, -1, 4])


def exercise2_6():
    return np.array([1, 3]), np.array([[-1, -1], [-1, 1], [1, 2]]), np.array([-3, -1, 2])


def exercise2_7():
    return np.array([1, 3]), np.array([[-1, -1], [-1, 1], [-1, 2]]), np.array([-3, -1, 2])


class Test(TestCase):
    def test_lp_solve(self):
        verbose = True
        c = np.array([5, 4, 3])
        a = np.array([[2, 3, 1],
                      [4, 1, 2],
                      [3, 4, 2]])
        b = np.array([5, 11, 8])
        if verbose:
            print(dictionary.Dictionary(c, a, b))
        expected_res = LPResult.OPTIMAL
        expected_d = """ z = 13 -  1*x4 -  3*x2 -  1*x6
x1 =  2 -  2*x4 -  2*x2 +  1*x6
x5 =  1 +  2*x4 +  5*x2 -  0*x6
x3 =  1 +  3*x4 +  1*x2 -  2*x6"""
        res, d = lp_solve(c, a, b, verbose=verbose)
        self.assertEqual(expected_res, res)
        self.assertEqual(expected_d, d.__str__())

    def test_lp_solve_integer(self):
        verbose = False
        c, a, b = example1()
        expected_res = LPResult.OPTIMAL
        expected_d = """ z = 13 -  1*x4 -  3*x2 -  1*x6
x1 =  2 -  2*x4 -  2*x2 +  1*x6
x5 =  1 +  2*x4 +  5*x2 -  0*x6
x3 =  1 +  3*x4 +  1*x2 -  2*x6"""
        res, d = lp_solve(c, a, b, int, verbose=verbose)
        self.assertEqual(expected_res, res)
        self.assertEqual(expected_d, d.__str__())

    def test_lp_solve2(self):
        verbose = False
        c = np.array([5, 2])
        a = np.array([[3, 1],
                      [2, 5]])
        b = np.array([7, 5])
        expected_res = LPResult.OPTIMAL
        expected_d = """ z = 152/13 -  21/13*x3 -   1/13*x4
x1 =  30/13 -   5/13*x3 +   1/13*x4
x2 =   1/13 +   2/13*x3 -   3/13*x4"""
        res, d = lp_solve(c, a, b, verbose=verbose)
        self.assertEqual(expected_res, res)
        self.assertEqual(expected_d, d.__str__())

    def test_lp_solve_integer2(self):
        verbose = False
        c = np.array([5, 2])
        a = np.array([[3, 1],
                      [2, 5]])
        b = np.array([7, 5])
        expected_res = LPResult.OPTIMAL
        expected_d = """13* z = 152 -  21*x3 -   1*x4
13*x1 =  30 -   5*x3 +   1*x4
13*x2 =   1 +   2*x3 -   3*x4"""
        res, d = lp_solve(c, a, b, int, verbose=verbose)
        self.assertEqual(expected_res, res)
        self.assertEqual(expected_d, d.__str__())

    def test_given_examples_fraction_2(self):
        verbose = False
        c, a, b = example2()
        if verbose:
            initial_d = Dictionary(c, a, b)
            print("Initial dictionary:")
            print(initial_d)
            print("------------------------")
        expected_res = LPResult.OPTIMAL
        expected_d = """ z = -8/3 -  4/3*x3 -  5/3*x4
x2 =  1/3 -  1/3*x3 +  1/3*x4
x1 =  4/3 +  2/3*x3 +  1/3*x4
x5 =  2/3 +  1/3*x3 -  1/3*x4"""
        res, d = lp_solve(c, a, b, verbose=verbose)
        self.assertEqual(expected_res, res)
        self.assertEqual(expected_d, d.__str__())

    def test_given_examples_integer_3(self):
        verbose = False
        c, a, b = integer_pivoting_example()
        if verbose:
            initial_d = Dictionary(c, a, b)
            print("Initial dictionary:")
            print(initial_d)
            print("------------------------")
        expected_res = LPResult.OPTIMAL
        expected_d = """13* z = 152 -  21*x3 -   1*x4
13*x1 =  30 -   5*x3 +   1*x4
13*x2 =   1 +   2*x3 -   3*x4"""
        res, d = lp_solve(c, a, b, int, verbose=verbose)
        self.assertEqual(expected_res, res)
        self.assertEqual(expected_d, d.__str__())

    def test_given_examples_integer_4(self):
        # TODO - test fails - File "C:\Users\sx5\PycharmProjects\optimization\lpsolve.py", line 204, in lp_solve_two_phase
        #     aggregate += v3
        # TODO: solve and compare to linprog solution
        # numpy.core._exceptions._UFuncOutputCastingError: Cannot cast ufunc 'add' output from dtype('O') to dtype('int32') with casting rule 'same_kind'
        verbose = False
        c, a, b = exercise2_5()
        if verbose:
            initial_d = Dictionary(c, a, b)
            print("Initial dictionary:")
            print(initial_d)
            print("------------------------")
        print(linprog(-c, a, b))
        expected_res = LPResult.OPTIMAL
        res, d = lp_solve(c, a, b, np.float64)
        self.assertEqual(expected_res, res)
        res, d = lp_solve(c, a, b, int)
        self.assertEqual(expected_res, res)

    def test_given_examples_integer_5(self):
        verbose = False
        c, a, b = exercise2_6()
        if verbose:
            initial_d = Dictionary(c, a, b)
            print("Initial dictionary:")
            print(initial_d)
            print("------------------------")
        expected_res = LPResult.INFEASIBLE
        res, d = lp_solve(c, a, b, int, verbose=verbose)
        self.assertEqual(expected_res, res)

    def test_given_examples_integer_6(self):
        verbose = False
        c, a, b = exercise2_7()
        if verbose:
            initial_d = Dictionary(c, a, b)
            print("Initial dictionary:")
            print(initial_d)
            print("------------------------")
        expected_res = LPResult.OPTIMAL
        print(linprog(-c, a, b))
        # TODO: The problem is unbounded but we return infeasible
        expected_d = """"""
        res, d = lp_solve(c, a, b, int, verbose=verbose)
        self.assertEqual(expected_res, res)
        self.assertEqual(expected_d, d.__str__())

    def test_two_phase_float_1(self):
        verbose = False
        c = np.array([1, -1, 1])
        a = np.array([[2, -3, 1], [2, -1, 2], [-1, 1, -2]])
        b = np.array([-5, 4, -1])
        if verbose:
            initial_d = Dictionary(c, a, b)
            print("Initial dictionary:")
            print(initial_d)
        expected_res = LPResult.OPTIMAL
        expected_d = """ z =  3/5 -  2/5*x5 -  1/5*x1 -  1/5*x4
x3 = 17/5 -  3/5*x5 -  4/5*x1 +  1/5*x4
x6 =    3 -    1*x5 -    1*x1 -    0*x4
x2 = 14/5 -  1/5*x5 +  2/5*x1 +  2/5*x4"""
        res, d = lp_solve(c, a, b, verbose=verbose)
        self.assertEqual(expected_res, res)
        self.assertEqual(expected_d, d.__str__())
        print(f"d.value(): {d.value()}")
        print(f"d.basic_solution: {d.basic_solution()}")
        print("Expected results")
        res_linprog = lpsolve.linprog(c, a, b)
        print(res_linprog)

    def test_try_linprog(self):
        # Works directly on auxiliary
        c = np.array([0, 0, 0, -1])
        a = np.array([[2, -3, 1, -1], [2, -1, 2, -1], [-1, 1, -2, -1]])
        b = np.array([-5, 4, -1])
        a_eq = None
        b_eq = None
        res_linprog = lpsolve.linprog(c, a, b, a_eq, b_eq)
        print(res_linprog)
        print()
        # On modified auxiliary
        c = np.array([-2, 3, -1, -1, -1])
        a = np.array([[-2, 3, -1, -1, 0], [0, 2, 1, -1, 0], [-3, 4, -3, -1, 0]])
        b = np.array([5, 9, 4])
        a_eq = np.array([[0, 0, 0, 0, 1]])
        b_eq = np.array([5])
        res_linprog = lpsolve.linprog(c, a, b, a_eq, b_eq)
        print(res_linprog)



"""
res: OptimizeResult
A scipy.optimize.OptimizeResult consisting of the fields:

x1-D array
The values of the decision variables that minimizes the objective function while satisfying the constraints.

fun : float
The optimal value of the objective function c @ x.

slack : 1-D array
The (nominally positive) values of the slack variables, b_ub - A_ub @ x.

con : 1-D array
The (nominally zero) residuals of the equality constraints, b_eq - A_eq @ x.

success : bool
True when the algorithm succeeds in finding an optimal solution.

status : int
An integer representing the exit status of the algorithm.

0 : Optimization terminated successfully.

1 : Iteration limit reached.

2 : Problem appears to be infeasible.

3 : Problem appears to be unbounded.

4 : Numerical difficulties encountered.

message : str
A string descriptor of the exit status of the algorithm.

nit : int
The total number of iterations performed in all phases.
"""
