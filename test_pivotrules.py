from unittest import TestCase

import numpy as np

from dictionary import Dictionary
from pivotrules import leaving_variable
from pivotrules import bland
from pivotrules import largest_coefficient
from pivotrules import largest_increase


class Test(TestCase):
    def test_leaving_variable(self):
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
        # Test leaving_variable for entering 0
        entering = 0
        expected = 2
        leaving = leaving_variable(d, 0, entering, verbose)
        self.assertEqual(expected, leaving)
        # Test leaving_variable for entering 1
        entering = 1
        expected = 1
        leaving = leaving_variable(d, 0, entering, verbose)
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
        leaving = leaving_variable(d, 0, entering, verbose)
        self.assertEqual(expected, leaving)
        # Test leaving_variable for entering 1
        entering = 1
        expected = 0
        leaving = leaving_variable(d, 0, entering, verbose)
        self.assertEqual(expected, leaving)
        # Test leaving_variable for entering 2
        entering = 2
        expected = 2
        leaving = leaving_variable(d, 0, entering, verbose)
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
        verbose = True
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

    def test_largest_increase(self):
        raise NotImplementedError


