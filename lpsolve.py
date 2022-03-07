from fractions import Fraction

import numpy as np

import dictionary
from dictionary import Dictionary
from lpresult import LPResult
from pivotrules import bland, eps_correction
from scipy.optimize import linprog as linprog_original


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
    return simple_simplex(c, a, b, dtype, eps, pivotrule, verbose)

    # lp_solve_two_phase(c, A, b, dtype, eps, pivotrule, verbose)


def simple_simplex(c, a, b, dtype=Fraction, eps=0, pivotrule=lambda D: bland(D, eps=0), verbose=False):
    d = Dictionary(c, a, b, dtype)
    return simplex(d, eps, pivotrule, verbose)


def simplex(d, eps=0, pivotrule=lambda D: bland(D, eps=0), verbose=False):
    degenerate_steps_before_anti_cycle = 10
    if (d.C[1:, 0] < 0).any():  # infeasible (Without auxiliary)
        if verbose:
            print("Simplex is infeasible")
        # TODO: Remove the prints below, once you have figured our the auxiliary infeasibility check problemo
        print("Simplex is infeasible")
        res_linprog = linprog(*dictionary_to_input_arrays(d))
        print("linprog (With value shown with opposite sign)")
        print(res_linprog)
        return LPResult.INFEASIBLE, None

    degenerate_counter = 0
    entering, leaving = pivotrule(d, eps)
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
                    pivotrule = lambda d, eps: bland(d, eps)
                break
        else:
            degenerate_counter = 0
        entering, leaving = pivotrule(d, eps)

    if entering is not None and leaving is None:
        if verbose:
            print(f"Simplex is unbounded")
        return LPResult.UNBOUNDED, None
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
    if result_auxiliary != LPResult.OPTIMAL:
        return LPResult.INFEASIBLE, None
    return phase_two(d_after_phase_one, c, a, b, dtype, eps, pivotrule, verbose)


def phase_two(d_auxiliary, c, a, b, dtype, eps=0, pivotrule=lambda D: bland(D, eps=0), verbose=False):
    # TODO: method for doing pivots on original to prepare for simplex
    # TODO: run simplex and return result
    raise NotImplementedError


def phase_one(c, a, b, dtype, eps=0, pivotrule=lambda D: bland(D, eps=0), verbose=False):
    d = Dictionary(None, a, b, dtype)
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
    # if verbose:
    #     print(f"The value of the optimal solution for the auxiliary problem: {d.value()}")
    # if d.value() < 0:
    #     return LPResult.INFEASIBLE, d
    leaving = position_of_auxiliary_variable_in_basis(b, verbose)
    if entering is not None:
        b.pivot(0, leaving)
    return LPResult.OPTIMAL, b


def position_of_auxiliary_variable_in_basis(d: dictionary.Dictionary, verbose):
    # TODO
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


def linprog(c, a_ub=None, b_ub=None, a_eq=None, b_eq=None):
    res = linprog_original(-c,
                           A_ub=a_ub,
                           b_ub=b_ub,
                           A_eq=a_eq,
                           b_eq=b_eq,
                           method='simplex')  # We need to say that the optimization should use the simplex method
    return res


def dictionary_to_input_arrays(d, verbose=False):
    # c, a_ub, b_ub = np.array([5, 4, 3]), np.array([[2, 3, 1], [4, 1, 2], [3, 4, 2]]), np.array([5, 11, 8])
    # d = dictionary.Dictionary(c, a_ub, b_ub)
    # print(d)
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
        return c, a_ub, b_ub, None, None
    else:
        a_row, a_col = a_ub.shape
        if isinstance(objective_function_constant, Fraction):
            sign = np.sign(objective_function_constant.numerator)
            c_plus = np.hstack([c, [Fraction(sign, 1)]])
            a_ub_plus = np.hstack([a_ub, np.full((a_row, 1), Fraction(0, 1))])
            a_eq = np.full((1, a_col + 1), Fraction(0, 1))
            a_eq[0, a_col] = Fraction(1, 1)
        else:
            sign = np.sign(objective_function_constant)
            c_plus = np.hstack([c, sign])
            a_ub_plus = np.hstack([a_ub, np.full((a_row, 1), 0)])
            a_eq = np.full((1, a_col + 1), 0)
            a_eq[0, a_col] = 1
        b_eq = np.array([sign * objective_function_constant])
        if verbose:
            print("Dictionary as arrays")
            print(f"c: {c_plus}")
            print(f"a_ub: {a_ub_plus}")
            print(f"b_ub: {b_ub}")
            print(f"a_eq: {a_eq}")
            print(f"b_eq: {b_eq}")
        return c_plus, a_ub_plus, b_ub, a_eq, b_eq

