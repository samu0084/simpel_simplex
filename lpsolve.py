from fractions import Fraction

import numpy as np

import dictionary
from dictionary import Dictionary
from lpresult import LPResult
from pivotrules import bland, eps_correction


def lp_solve(c, a, b, dtype=Fraction, eps=0, pivotrule=lambda D: bland(D, eps=0), verbose=False):
    # Simplex algorithm
    #
    # Input is LP in standard form given by vectors and matrices
    # c,A,b.
    #
    # eps>=0 is such that numbers in the closed interval [-eps,eps]
    # are to be treated as if they were 0.
    #
    # pivotrule is a rule used for pivoting. Cycling is prevented by
    # switching to Bland's rule as needed.
    #
    # If verbose is True it outputs possible useful information about
    # the execution, e.g. the sequence of pivot operations
    # performed. Nothing is required.
    #
    # If LP is infeasible the return value is LPResult.INFEASIBLE,None
    #
    # If LP is unbounded the return value is LPResult.UNBOUNDED,None
    #
    # If LP has an optimal solution the return value is
    # LPResult.OPTIMAL,d, where d is an optimal dictionary.
    return simple_simplex(c, a, b, eps, pivotrule, verbose)

    # lp_solve_two_phase(c, A, b, dtype, eps, pivotrule, verbose)


def simple_simplex(c, a, b, dtype=Fraction, eps=0, pivotrule=lambda D: bland(D, eps=0), verbose=False):
    d = Dictionary(c, a, b, dtype)
    return simplex(d, eps, pivotrule, verbose)


def simplex(d, eps=0, pivotrule=lambda D: bland(D, eps=0), verbose=False):
    degenerate_steps_before_anti_cycle = 10
    if not (d.basic_solution() == np.zeros(d.N.shape[0])).all():  # infeasible (Without auxiliary)
        print("Simplex is infeasible")
        return LPResult.INFEASIBLE, None

    degenerate_counter = 0
    entering, leaving = pivotrule(d)
    while entering is not None and leaving is not None:
        if verbose:
            print(f"entering: {d.varnames[d.N[entering]]}; leaving: {d.varnames[d.B[leaving]]}")
        d.pivot(entering, leaving)
        if verbose:
            print(f"After pivot ({entering, leaving})")
            print(d)
            print(f"Check if there are any zero constants: {d.C[:, 0][1:]}")
            print(f"Compare with {np.zeros(d.C.shape[0] - 1)}")
        # Degenerate steps check
        for const in d.C[:, 0][1:]:
            if eps_correction(const, eps) == 0:  # New dictionary is degenerate
                degenerate_counter += 1
                #
                if degenerate_counter > degenerate_steps_before_anti_cycle:
                    pivotrule = bland
                break
        else:
            degenerate_counter = 0
        entering, leaving = pivotrule(d)

    if entering is not None and leaving is None:
        print(f"Simplex is unbounded")
        return LPResult.UNBOUNDED, None
    print(f"Type returned from simplex: {type(d)}")
    return LPResult.OPTIMAL, d


def lp_solve_two_phase(c, a, b, dtype=Fraction, eps=0, pivotrule=lambda D: bland(D, eps=0), verbose=False):
    if (b >= 0).all():
        if verbose:
            print("All constants are greater then zero. We run the simple simplex")
        return simple_simplex(c, a, b, dtype, eps, pivotrule, verbose)
    # Else we run the two phase simplex
    result_auxiliary, d_after_phase_one = phase_one(c, a, b, dtype, eps, pivotrule, verbose)
    if verbose:
        print(f"result auxiliary: {result_auxiliary}")
        print(f"Dictionary after phase one:")
        print(d_after_phase_one)
    # TODO: method for doing pivots on original to prepare for simplex
    # TODO: run simplex and return result
    raise NotImplementedError


def phase_one(c, a, b, dtype, eps=0, pivotrule=lambda D: bland(D, eps=0), verbose=False):
    d = Dictionary(None, a, b, dtype)
    print(f"Type initial: {type(d)}")
    if verbose:
        print(f"Auxiliary initial dictionary:")
        print(d)
    entering = d.N.shape[0] - 1
    leaving = lowest_constraint_const(d, verbose)
    if verbose:
        print(f"Pivot variable from lowest_constraint_const(...): {leaving}")
        print(f"entering: {entering}")
        print(f"leaving: {leaving}")
    d.pivot(entering, leaving, verbose)
    if verbose:
        print(f"Auxiliary dictionary after pivot with entering = {entering} and leaving = {leaving}")
        print(d)
    result, b = simplex(d, eps, pivotrule, verbose)
    print(f"Type after simplex: {type(b)}")
    # if verbose:
    #     print(f"The value of the optimal solution for the auxiliary problem: {d.value()}")
    # if d.value() < 0:
    #     return LPResult.INFEASIBLE, d
    leaving = position_of_auxiliary_variable_in_basis(b, verbose)
    if entering is not None:
        b.pivot(0, leaving)
    return LPResult.OPTIMAL, b


def position_of_auxiliary_variable_in_basis(d: dictionary.Dictionary, verbose):
    print(f"Type in method: {type(d)}")
    # print(f"varnames: {d.varnames[0]}")
    #    print(f"position_of_auxiliary_variable_in_basis: {}")
    return None


def lowest_constraint_const(d, verbose=False):
    cols = d.C.shape[0]
    lowest_const = 0
    basic_variable = 0
    for column in range(1, cols):
        if lowest_const > d.C[column, 0]:
            lowest_const = d.C[column, 0]
            basic_variable = column - 1
        if verbose:
            print(f"Constraint: {column}); Contains constant: {d.C[column, 0]}; "
                  f"Lowest constant until now: {lowest_const}; Basic variable number: {basic_variable}")
    return basic_variable
