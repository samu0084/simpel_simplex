from fractions import Fraction
from unittest import TestCase

import numpy as np

import dictionary
from dictionary import Dictionary
from lpresult import LPResult
from lpsolve import lp_solve, lp_solve_two_phase
from scipy.optimize import linprog as linprog_original


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


def linprog(c, a, b):
    res = linprog_original(-c,
                  A_ub=a,
                  b_ub=b,
                  method='simplex')  # We need to explicitly say that the optimization should use the simplex method
    return res


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
        # TODO
        expected_d = """"""
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
        # TODO
        expected_d = """"""
        res, d = lp_solve(c, a, b, int, verbose=verbose)
        self.assertEqual(expected_res, res)
        self.assertEqual(expected_d, d.__str__())

    def test_given_examples_integer_4(self):
        verbose = False
        c, a, b = exercise2_5()
        if verbose:
            initial_d = Dictionary(c, a, b)
            print("Initial dictionary:")
            print(initial_d)
            print("------------------------")
        expected_res = LPResult.OPTIMAL
        # TODO
        expected_d = """"""
        res, d = lp_solve(c, a, b, int, verbose=verbose)
        self.assertEqual(expected_res, res)
        self.assertEqual(expected_d, d.__str__())

    def test_given_examples_integer_5(self):
        verbose = False
        c, a, b = exercise2_6()
        if verbose:
            initial_d = Dictionary(c, a, b)
            print("Initial dictionary:")
            print(initial_d)
            print("------------------------")
        expected_res = LPResult.OPTIMAL
        # TODO
        expected_d = """"""
        res, d = lp_solve(c, a, b, int, verbose=verbose)
        self.assertEqual(expected_res, res)
        self.assertEqual(expected_d, d.__str__())

    def test_given_examples_integer_5(self):
        verbose = False
        c, a, b = exercise2_7()
        if verbose:
            initial_d = Dictionary(c, a, b)
            print("Initial dictionary:")
            print(initial_d)
            print("------------------------")
        expected_res = LPResult.OPTIMAL
        # TODO
        expected_d = """"""
        res, d = lp_solve(c, a, b, int, verbose=verbose)
        self.assertEqual(expected_res, res)
        self.assertEqual(expected_d, d.__str__())

    def test_two_phase_float_1(self):
        verbose = True
        c = np.array([1, -1, 1])
        a = np.array([[2, -3, 1], [2, -1, 2], [-1, 1, -2]])
        b = np.array([-5, 4, -1])
        res_linprog = linprog(c, a, b)
        print("linprog:")
        print(f"Value: {-res_linprog.fun}")
        if verbose:
            initial_d = Dictionary(c, a, b)
            print("Initial dictionary:")
            print(initial_d)
            print("------------------------")
        expected_res = LPResult.OPTIMAL
        expected_d = """"""
        res, d = lp_solve_two_phase(c, a, b, verbose=verbose)
        self.assertEqual(expected_res, res)
        self.assertEqual(expected_d, d.__str__())

    def test_dictionary_to_input_arrays(self, d, verbose):
        c, a_ub, b_ub = np.array([5, 4, 3]), np.array([[2, 3, 1], [4, 1, 2], [3, 4, 2]]), np.array([5, 11, 8])
        d = dictionary.Dictionary(c, a_ub, b_ub)
        print(d)
        c = d.C[0, 1:]
        a_ub = -d.C[1:, 1:]
        b_ub = d.C[1:, 0]
        objective_function_constant = d.C[0, 0]
        if objective_function_constant == 0:
            if verbose:
                print("Dictionary as arrays")
                print(f"c: {c}")
                print(f"a_ub: {a_ub}")
                print(f"b_ub: {b_ub}")
            return c, a_ub, b_ub
        else:
            c_plus = np.hstack([c, [Fraction(1, 1)]])
            a_row, a_col = a_ub.shape
            a_ub_plus = np.hstack([a_ub, np.full((a_row, 1), Fraction(0, 1))])
            a_eq = np.full((a_col + 1), Fraction(0, 1))
            a_eq[a_col] = Fraction(1, 1)
            b_eq = np.array([objective_function_constant])
            if verbose:
                print("Dictionary as arrays")
                print(f"c: {c_plus}")
                print(f"a_ub: {a_ub_plus}")
                print(f"b_ub: {b_ub}")
                print(f"a_eq: {a_eq}")
                print(f"b_eq: {b_eq}")
            return c_plus, a_row, a_ub_plus, a_eq, b_eq

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
