from unittest import TestCase

import numpy as np

from dictionary import Dictionary


class TestDictionary(TestCase):
    def test_float_pivot(self):
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
        # Test pivot: entering x1, leaving x5
        expected = """ z = 56 -  7*x5 +  4*x2
x3 =  4 +  2*x5 -  1*x2
x4 = 10 +  1*x5 -  1*x2
x1 =  8 -  1*x5 -  0*x2"""
        d.pivot(0, 2)  # (first non-basic; third basic)
        if verbose:
            print(d)
            print()
        self.assertTrue(d.__str__() == expected)

        # Test pivot: entering x2, leaving x3
        expected = """ z = 72 +  1*x5 -  4*x3
x2 =  4 +  2*x5 -  1*x3
x4 =  6 -  1*x5 +  1*x3
x1 =  8 -  1*x5 -  0*x3"""
        d.pivot(1, 0)  # (second non-basic; first basic)
        if verbose:
            print(d)
            print()
        self.assertTrue(d.__str__() == expected)

        # Test pivot: entering x5, leaving x4
        expected = """ z = 78 -  1*x4 -  3*x3
x2 = 16 -  2*x4 +  1*x3
x5 =  6 -  1*x4 +  1*x3
x1 =  2 +  1*x4 -  1*x3"""
        d.pivot(0, 1)  # (first non-basic; second basic)
        if verbose:
            print(d)
            print()
        self.assertTrue(d.__str__() == expected)

    def test_float_pivot2(self):
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
        # Test pivot: entering x1, leaving x4
        expected = """ z = 25/2 -  5/2*x4 -  7/2*x2 +  1/2*x3
x1 =  5/2 -  1/2*x4 -  3/2*x2 -  1/2*x3
x5 =    1 +    2*x4 +    5*x2 -    0*x3
x6 =  1/2 +  3/2*x4 +  1/2*x2 -  1/2*x3"""
        d.pivot(0, 0, verbose)  # (first non-basic e.g. x1; first basic e.g. x4)
        if verbose:
            print("After pivot(0, 0)")
            print(d)
            print()
        self.assertTrue(d.__str__() == expected)

        # Test pivot: entering x3, leaving x6
        expected = """ z = 13 -  1*x4 -  3*x2 -  1*x6
x1 =  2 -  2*x4 -  2*x2 +  1*x6
x5 =  1 +  2*x4 +  5*x2 -  0*x6
x3 =  1 +  3*x4 +  1*x2 -  2*x6"""
        d.pivot(2, 2, verbose)  # (second non-basic; second basic)
        if verbose:
            print("After pivot(2, 2)")
            print(d)
            print()
        self.assertTrue(d.__str__() == expected)

    def test_float_pivot3(self):
        verbose = False
        # Make dictionary
        c = np.array([5, 4, 3])
        a = np.array([[2, 3, 1],
                      [4, 1, 2],
                      [3, 4, 2]])
        b = np.array([5, 11, 8])
        d = Dictionary(c, a, b)
        if verbose:
            print("Before pivot")
            print(d)
            print()
        # Test pivot which would not occur in algorithm: entering x2, leaving x6
        expected = """ z =    8 +    2*x1 -    1*x6 +    1*x3
x4 =   -1 +  1/4*x1 +  3/4*x6 +  1/2*x3
x5 =    9 - 13/4*x1 +  1/4*x6 -  3/2*x3
x2 =    2 -  3/4*x1 -  1/4*x6 -  1/2*x3"""
        d.pivot(1, 2, verbose)  # (second non-basic; third basic)
        if verbose:
            print("After pivot(1, 2)")
            print(d)
            print()
        self.assertTrue(d.__str__() == expected)

    def test_integer_pivot(self):
        verbose = False
        # Make dictionary
        c = np.array([5, 2])
        a = np.array([[3, 1],
                      [2, 5]])
        b = np.array([7, 5])
        d = Dictionary(c, a, b, int)
        if verbose:
            print(d)
            print()
        # Test pivot: entering x1, leaving x5
        expected = """3* z = 35 -  5*x3 +  1*x2
3*x1 =  7 -  1*x3 -  1*x2
3*x4 =  1 +  2*x3 - 13*x2"""
        d.pivot(0, 0, verbose)  # (first non-basic; third basic)
        self.assertTrue(d.__str__() == expected)
        # Test pivot: entering x2, leaving x3
        expected = """13* z = 152 -  21*x3 -   1*x4
13*x1 =  30 -   5*x3 +   1*x4
13*x2 =   1 +   2*x3 -   3*x4"""
        d.pivot(1, 1, verbose)  # (second non-basic; first basic)
        if verbose:
            print(d)
            print()
        self.assertTrue(d.__str__() == expected)


"""
def custom1():
    return np.array([4,6]),np.array([[2,-2],[4,0]]),np.array([6,16])
def example1():
    return np.array([5,4,3]),np.array([[2,3,1],[4,1,2],[3,4,2]]),np.array([5,11,8])
def example2():
    return np.array([-2,-1]),np.array([[-1,1],[-1,-2],[0,1]]),np.array([-1,-2,1])
def integer_pivoting_example():
    return np.array([5,2]),np.array([[3,1],[2,5]]),np.array([7,5])
def exercise2_5():
    return np.array([1,3]),np.array([[-1,-1],[-1,1],[1,2]]),np.array([-3,-1,4])
def exercise2_6():
    return np.array([1,3]),np.array([[-1,-1],[-1,1],[1,2]]),np.array([-3,-1,2])
def exercise2_7():
    return np.array([1,3]),np.array([[-1,-1],[-1,1],[-1,2]]),np.array([-3,-1,2])
def random_lp(n,m,sigma=10):
    return np.round(sigma*np.random.randn(n)),np.round(sigma*np.random.randn(m,n)),np.round(sigma*np.abs(np.random.randn(m)))
    
    
def run_examples(first, to):
    if first <= 1 and to >= 1:
        # Example 1
        c,A,b = example1()
        D=Dictionary(c,A,b)
        print('Example 1 with Fraction')
        print('Initial dictionary:')
        print(D)
        
        print('x1 is entering and x4 leaving:')
        D.pivot(0,0)
        print(D)
        print('x3 is entering and x6 leaving:')
        D.pivot(2,2)
        print(D)
        print()
     
    if first <= 2 and to >= 2:
        c,A,b = example1()
        D=Dictionary(c,A,b,np.float64)
        print('Example 1 with np.float64')
        print('Initial dictionary:')
        print(D)
        print('x1 is entering and x4 leaving:')
        D.pivot(0,0)
        print(D)
        print('x3 is entering and x6 leaving:')
        D.pivot(2,2)
        print(D)
        print()
    
    if first <= 3 and to >= 3:
        # Example 2
        c,A,b = example2()
        print('Example 2')
        print('Auxillary dictionary')
        D=Dictionary(None,A,b)
        print(D)
        print('x0 is entering and x4 leaving:')
        D.pivot(2,1)
        print(D)
        print('x2 is entering and x3 leaving:')
        D.pivot(1,0)
        print(D)
        print('x1 is entering and x0 leaving:')
        D.pivot(0,1)
        print(D)
        print()
        c,A,b = example2()
        res,D=lp_solve(None,A,b)
        print(res)
        print(D)
        print()

    if first <= 4 and to >= 4:
        # Solve Example 1 using lp_solve
        c,A,b = example1()
        print('lp_solve Example 1:')
        res,D=lp_solve(c,A,b)
        print(res)
        print(D)
        print()

    if first <= 5 and to >= 5:
        # Solve Example 2 using lp_solve
        c,A,b = example2()
        print('lp_solve Example 2:')
        res,D=lp_solve(None,A,b)
        print(res)
        print(D)
        print()

    if first <= 6 and to >= 6:
        # Solve Exercise 2.5 using lp_solve
        c,A,b = exercise2_5()
        print('lp_solve Exercise 2.5:')
        res,D=lp_solve(c,A,b)
        print(res)
        print(D)
        print()

    if first <= 7 and to >= 7:
        # Solve Exercise 2.6 using lp_solve
        c,A,b = exercise2_6()
        print('lp_solve Exercise 2.6:')
        res,D=lp_solve(c,A,b)
        print(res)
        print(D)
        print()

    if first <= 8 and to >= 8:
        # Solve Exercise 2.7 using lp_solve
        c,A,b = exercise2_7()
        print('lp_solve Exercise 2.7:')
        res,D=lp_solve(c,A,b)
        print(res)
        print(D)
        print()

    if first <= 9 and to >= 9:
        #Integer pivoting
        c,A,b=example1()
        D=Dictionary(c,A,b,int)
        print('Example 1 with int')
        print('Initial dictionary:')
        print(D)
        print('x1 is entering and x4 leaving:')
        D.pivot(0,0)
        print(D)
        print('x3 is entering and x6 leaving:')
        D.pivot(2,2)
        print(D)
        print()

    if first <= 10 and to >= 10:
        c,A,b = integer_pivoting_example()
        D=Dictionary(c,A,b,int)
        print('Integer pivoting example from lecture')
        print('Initial dictionary:')
        print(D)
        print('x1 is entering and x3 leaving:')
        D.pivot(0,0)
        print(D)
        print('x2 is entering and x4 leaving:')
        D.pivot(1,1)
        print(D)  
"""
