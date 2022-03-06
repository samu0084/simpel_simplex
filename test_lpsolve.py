from fractions import Fraction
from unittest import TestCase

import numpy as np

import dictionary
from dictionary import Dictionary
from lpresult import LPResult
from lpsolve import lp_solve, lp_solve_two_phase


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


