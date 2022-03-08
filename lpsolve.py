from fractions import Fraction

import numpy as np

import dictionary
from dictionary import Dictionary
from lpresult import LPResult
from pivotrules import bland, eps_correction
from scipy.optimize import linprog as linprog_original
import scipy.optimize


def lp_solve(c, a, b, dtype=Fraction, eps=0, pivotrule=lambda d, eps: bland(d, eps=0), verbose=False):
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
    return lp_solve_two_phase(c, a, b, dtype, eps, pivotrule, verbose)

    # lp_solve_two_phase(c, A, b, dtype, eps, pivotrule, verbose)


def simple_simplex(c, a, b, dtype=Fraction, eps=0, pivotrule=lambda D: bland(D, eps=0), verbose=False):
    d = Dictionary(c, a, b, dtype)
    return simplex(d, eps, pivotrule, verbose)


"""
#check if bounds are negative
    min_b = np.argmin(b)
    if b[min_b]<0:
        #PHASE 1 OF auxilary method
       prevous_z = c
       prevous_N = D.N+1
       most_infeasible = min_b
       new_z = np.zeros(len(c))
       if dtype== fraction:
           new_c = np.insert(new_z,0,fraction(-1))
       else:
           new_c = np.insert(new_z, 0,-1)
       new_A = np.insert(A,0,-1,axis=1)
       D=Dictionary(new_c,new_A,b,dtype=dtype)
       D.varnames = np.insert(D.varnames[:-1],1,"x0")
       D.pivot(0,most_infeasible)
       bs = D.C[1:0]
       #still still ifeasible find enter and leaving varaible.
      # and pivot.
      #if optimal solution to the auxilary problem is not zero, the orginal problem is infeasible.
     # phase 2
      #remove x_0 from dictionary
      # remove x_0 from non-bascis
     # calculate new cost function
     #lastly create a new dictionary without x0
"""


def simplex(d, eps=0, pivotrule=lambda d, eps: bland(d, eps=0), verbose=False):
    degenerate_steps_before_anti_cycle = 10
    if (d.C[1:, 0] < 0).any():  # infeasible (Without auxiliary)
        if verbose:
            print("Simplex is infeasible")
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
            if verbose:
                print(f"There where no zero constants.")
        entering, leaving = pivotrule(d, eps)

    if entering is not None and leaving is None:
        if verbose:
            print(f"Simplex is unbounded")
        return LPResult.UNBOUNDED, None
    return LPResult.OPTIMAL, d


def lp_solve_two_phase(c, a, b, dtype=Fraction, eps=0, pivotrule=lambda d, eps: bland(d, eps), verbose=False):
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


def phase_one(c, a, b, dtype, eps=0, pivotrule=lambda d, eps: bland(d, eps=0), verbose=False):
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
    result, d = simplex(d, eps, pivotrule, verbose)
    if result == LPResult.OPTIMAL:
        is_auxiliary_variable_in_basis, position_in_basis = position_of_auxiliary_variable_in_basis(d, verbose)
        if is_auxiliary_variable_in_basis:
            d.pivot(0, position_in_basis, verbose)
    return result, d


def position_of_auxiliary_variable_in_basis(d: dictionary.Dictionary, verbose):
    # The auxiliary variable-name is at varname[n+1] (n+1 = cols-1 in C). Thus we look for this entry in the basis.
    rows, cols = d.C.shape
    if verbose:
        print(f"rows, cols = d.C.shape: rows = {rows}, cols = {cols}")
        print(f"for leaving, entry in enumerate(d.B):")
    for leaving, entry in enumerate(d.B):
        if verbose:
            print(f"leaving = {leaving}")
            print(f"entry = {entry}")
            print(f"if entry == cols + 1: {entry == cols - 1}")
        if entry == cols - 1:
            if verbose:
                print(f"return True, {leaving}")
            return True, leaving
    if verbose:
        print("return False, None")
    return False, None


def basis_index_auxiliary_variable(d: dictionary.Dictionary, verbose):
    # The auxiliary variable-name is at varname[n+1] (n+1 = cols-1 in C). Thus we look for this entry in the basis.
    rows, cols = d.C.shape
    if verbose:
        print(f"rows, cols = d.C.shape: rows = {rows}, cols = {cols}")
    for index, entry in enumerate(d.N):
        if verbose:
            print(f"basis index = {index}")
            print(f"entry = {entry}")
            print(f"if entry == cols + 1: {entry == cols - 1}")
        if entry == cols - 1:
            if verbose:
                print(f"return {index}")
            return index
    if verbose:
        print("return None")
    return None


def push_elements_left(array, from_index):
    for i in range(from_index, 0):
        1


def phase_two(d_auxiliary, c, a, b, dtype, eps=0, pivotrule=lambda D: bland(D, eps=0), verbose=False):
    # identify and delete column of auxiliary variable
    basis_index_of_auxiliary_variable = basis_index_auxiliary_variable(d_auxiliary, verbose)
    d_auxiliary.N = np.delete(d_auxiliary.N, basis_index_of_auxiliary_variable, 0)
    d_auxiliary.C = np.delete(d_auxiliary.C, basis_index_of_auxiliary_variable + 1, 1)
    # correct OF
    d_auxiliary.C[0, 1:] = c
    aggregate = np.full((len(c)+1), 0, dtype)
    for index, b in enumerate(d_auxiliary.B):
        if b <= len(c):
            aggregate += d_auxiliary.C[index + 1, :] * c[b-1]
            d_auxiliary.C[0, b] = 0
    d_auxiliary.C[0, :] += aggregate
    if verbose:
        print("---------------------changed auxiliary-------------------------")
        print(d_auxiliary)
    return simplex(d_auxiliary, eps=eps, pivotrule=pivotrule, verbose=verbose)


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
