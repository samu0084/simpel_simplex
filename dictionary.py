import math

import numpy as np
from fractions import Fraction


class Dictionary:
    # Simplex dictionary as defined by Vanderbei
    #
    # 'C' is a (m+1)x(n+1) NumPy array that stores all the coefficients
    # of the dictionary.
    #
    # 'dtype' is the type of the entries of the dictionary. It is
    # supposed to be one of the native (full precision) Python types
    # 'int' or 'Fraction' or any Numpy type such as 'np.float64'.
    #
    # dtype 'int' is used for integer pivoting. Here an additional
    # variables 'lastpivot' is used. 'lastpivot' is the negative pivot
    # coefficient of the previous pivot operation. Dividing all
    # entries of the integer dictionary by 'lastpivot' results in the
    # normal dictionary.
    #
    # Variables are indexed from 0 to n+m. Variable 0 is the objective
    # z. Variables 1 to n are the original variables. Variables n+1 to
    # n+m are the slack variables. An exception is when creating an
    # auxillary dictionary where variable n+1 is the auxillary
    # variable (named x0) and variables n+2 to n+m+1 are the slack
    # variables (still names x{n+1} to x{n+m}).
    #
    # 'B' and 'N' are arrays that contain the *indices* of the basic and
    # nonbasic variables.
    #
    # 'varnames' is an array of the names of the variables.

    def __init__(self, c, A, b, dtype=Fraction):
        # Initializes the dictionary based on linear program in
        # standard form given by vectors and matrices 'c','A','b'.
        # Dimensions are inferred from 'A'
        #
        # If 'c' is None it generates the auxillary dictionary for the
        # use in the standard two-phase simplex algorithm
        #
        # Every entry of the input is individually converted to the
        # given dtype.
        m, n = A.shape
        self.dtype = dtype
        if dtype == int:
            self.lastpivot = 1
        if dtype in [int, Fraction]:
            dtype = object
            if c is not None:
                c = np.array(c, object)
            A = np.array(A, object)
            b = np.array(b, object)
        self.C = np.empty([m + 1, n + 1 + (c is None)], dtype=dtype)
        self.C[0, 0] = self.dtype(0)
        if c is None:
            self.C[0, 1:] = self.dtype(0)
            self.C[0, n + 1] = self.dtype(-1)
            self.C[1:, n + 1] = self.dtype(1)
        else:
            for j in range(0, n):
                self.C[0, j + 1] = self.dtype(c[j])
        for i in range(0, m):
            self.C[i + 1, 0] = self.dtype(b[i])
            for j in range(0, n):
                self.C[i + 1, j + 1] = self.dtype(-A[i, j])
        self.N = np.array(range(1, n + 1 + (c is None)))
        self.B = np.array(range(n + 1 + (c is None), n + 1 + (c is None) + m))
        self.varnames = np.empty(n + 1 + (c is None) + m, dtype=object)
        self.varnames[0] = 'z'
        for i in range(1, n + 1):
            self.varnames[i] = 'x{}'.format(i)
        if c is None:
            self.varnames[n + 1] = 'x0'
        for i in range(n + 1, n + m + 1):
            self.varnames[i + (c is None)] = 'x{}'.format(i)
        if self.dtype == int:
            self.basic_multiplier = 1

    def __str__(self):
        # String representation of the dictionary in equation form as
        # used in Vanderbei.
        m, n = self.C.shape
        varlen = len(max(self.varnames, key=len))
        coeflen = 0
        for i in range(0, m):
            coeflen = max(coeflen, len(str(self.C[i, 0])))
            for j in range(1, n):
                coeflen = max(coeflen, len(str(abs(self.C[i, j]))))
        tmp = []
        if self.dtype == int and self.lastpivot != 1:
            tmp.append(str(self.lastpivot))
            tmp.append('*')
        tmp.append('{} = '.format(self.varnames[0]).rjust(varlen + 3))
        tmp.append(str(self.C[0, 0]).rjust(coeflen))
        for j in range(0, n - 1):
            tmp.append(' + ' if self.C[0, j + 1] > 0 else ' - ')
            tmp.append(str(abs(self.C[0, j + 1])).rjust(coeflen))
            tmp.append('*')
            tmp.append('{}'.format(self.varnames[self.N[j]]).rjust(varlen))
        for i in range(0, m - 1):
            tmp.append('\n')
            if self.dtype == int and self.lastpivot != 1:
                tmp.append(str(self.lastpivot))
                tmp.append('*')
            tmp.append('{} = '.format(self.varnames[self.B[i]]).rjust(varlen + 3))
            tmp.append(str(self.C[i + 1, 0]).rjust(coeflen))
            for j in range(0, n - 1):
                tmp.append(' + ' if self.C[i + 1, j + 1] > 0 else ' - ')
                tmp.append(str(abs(self.C[i + 1, j + 1])).rjust(coeflen))
                tmp.append('*')
                tmp.append('{}'.format(self.varnames[self.N[j]]).rjust(varlen))
        return ''.join(tmp)

    def basic_solution(self):
        # Extracts the basic solution defined by a dictionary D
        m, n = self.C.shape
        if self.dtype == int:
            x_dtype = Fraction
        else:
            x_dtype = self.dtype
        x = np.empty(n - 1, x_dtype)
        x[:] = x_dtype(0)
        for i in range(0, m - 1):
            if self.B[i] < n:
                if self.dtype == int:
                    x[self.B[i] - 1] = Fraction(self.C[i + 1, 0], self.lastpivot)
                else:
                    x[self.B[i] - 1] = self.C[i + 1, 0]
        return x

    def value(self):
        # Extracts the value of the basic solution defined by a dictionary D
        if self.dtype == int:
            return Fraction(self.C[0, 0], self.lastpivot)
        else:
            return self.C[0, 0]

    def pivot(self, entering, leaving, verbose=False):
        # Pivot Dictionary with N[k] entering and B[l] leaving
        # Performs integer pivoting if self.dtype==int
        if self.dtype == int:
            self.integer_pivot(entering, leaving, verbose)
        else:
            self.float_pivot(entering, leaving, verbose)

    def integer_pivot(self, entering, leaving, verbose=False):
        a = self.C[leaving + 1, entering + 1]
        lastpivot = self.lastpivot
        if verbose:
            print(f"lastpivot: {lastpivot}")
            print(f"pivot coefficient: {a}")
            print(self)
        # Shift indices
        temp = self.N[entering]
        self.N[entering] = self.B[leaving]
        self.B[leaving] = temp
        # Multiply all rows, except pivot row, by the negative pivot coefficient
        for row in range(self.C.shape[0]):
            if row != leaving + 1:
                self.C[row, :] *= -a
        # Give new basis the multiplied value
        self.lastpivot = -a
        # Input value for previous basic
        self.C[leaving + 1, entering + 1] = -lastpivot
        for row in range(self.C.shape[0]):
            if row != leaving + 1:
                coefficient = self.C[row, entering + 1]
                pivot_row_multiplier = coefficient // -a
                self.C[row, entering + 1] = 0
                self.C[row, :] += pivot_row_multiplier * self.C[leaving + 1, :]
        if verbose:
            print("Step 2")
            print(self)
        # 	3. Divide all rows, except for pivot row, by the previous negative pivot coefficient
        for row in range(self.C.shape[0]):
            if row != leaving + 1:
                self.C[row, :] //= lastpivot
        if verbose:
            print("Step 3")
            print(self)
            print(f"self.lastpivot: {self.lastpivot}")

    def float_pivot(self, entering, leaving, verbose=False):
        if verbose:
            print(f"temp = self.N[entering]            temp = {self.N[entering]}")
            print(f"self.N[entering] = self.B[leaving] {self.N[entering]} = {self.B[leaving]}")
            print(f"self.B[leaving] = temp             {self.B[leaving]} = temp")
        # Shift indices
        temp = self.N[entering]
        self.N[entering] = self.B[leaving]
        self.B[leaving] = temp
        if verbose:
            print("Shift indices:")
            print(self)
        # Save pivot coefficient
        a = self.C[leaving + 1, entering + 1]
        if verbose:
            print("Save pivot coefficient:")
            print(f"{a} = self.C[l + 1, k + 1]")
        # Calculations on pivot row
        self.C[leaving + 1, :] /= -a
        self.C[leaving + 1, entering + 1] = 1 / a
        if verbose:
            print("Calculation on pivot row:")
            print(self)
        # Calculations on remaining rows
        for row in range(self.C.shape[0]):
            if row != leaving + 1:
                c = self.C[row, entering + 1]
                self.C[row, :] += c * self.C[leaving + 1, :]
                self.C[row, entering + 1] = c * self.C[leaving + 1, entering + 1]
        if verbose:
            print("Calculation on remaining rows:")
            print(self)

