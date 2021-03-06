from fractions import Fraction

import numpy as np

import dictionary
from dictionary import Dictionary
from lpresult import LPResult
from pivotrules import bland, eps_correction
from scipy.optimize import linprog as linprog_original
import scipy.optimize


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
#
# 1) Check if we can go directly to the simplex method (all constraint constants are greater than 0)
#       True: Construct dictionary
#             return simplex(...)
# 2) Construct auxiliary dictionary
# 3) Set the entering variable to be the auxiliary variable
# 4) Find the leaving variable, where the constraint have the numerically greatest negative constant
# 5) pivot(auxiliary variable, leaving identified above)
# 6) Use the simplex method in the dictionary
# 7) Check if simplex method is unbounded
#       True: return INFEASIBLE, None
# 8) Check if the auxiliary variable is in the basis
#       True: pivot the auxiliary variable out of the basis. The entering variable can be chosen arbitrarily
# 9) Identify location of auxiliary variable in the non-basis
# 10) Delete the column with the auxiliary variable, and
#     delete the element of N holding the auxiliary variable
# 11) Correct the OF
# 12) return simplex(...)
def lp_solve(c, a, b, dtype=Fraction, eps=0, pivotrule=lambda d, eps: bland(d, eps=0), verbose=False):
    if (b >= 0).all():
        d = dictionary.Dictionary(c, a, b, dtype)
        return simplex(d, eps, pivotrule)
    d_aux = Dictionary(None, a, b, dtype)
    entering = d_aux.N.shape[0] - 1
    leaving = lowest_constraint_const(d_aux)
    d_aux.pivot(entering, leaving)
    result_aux, d_aux = simplex(d_aux, eps, pivotrule)
    if result_aux != LPResult.OPTIMAL:
        return LPResult.INFEASIBLE, None
    is_auxiliary_variable_in_basis, position_in_basis = position_of_auxiliary_variable_in_basis(d_aux)
    if is_auxiliary_variable_in_basis:
        entering = 0
        d_aux.pivot(entering, position_in_basis)
    index_of_auxiliary_variable_in_basis = basis_index_auxiliary_variable(d_aux)
    d_aux.N = np.delete(d_aux.N, index_of_auxiliary_variable_in_basis, 0)
    d_aux.C = np.delete(d_aux.C, index_of_auxiliary_variable_in_basis + 1, 1)
    d_aux.C[0, 1:] = c
    aggregate = np.full((len(c) + 1), 0, dtype)
    for index, b in enumerate(d_aux.B):
        if b <= len(c):
            aggregate = aggregate + d_aux.C[index + 1, :] * c[b - 1]
            d_aux.C[0, b] = 0
    d_aux.C[0, :] += aggregate
    return simplex(d_aux, eps=eps, pivotrule=pivotrule)


# A simple wrapper method for the simplex algorithm which produces the dictionary and calls the simplex method.
def simple_simplex(c, a, b, dtype=Fraction, eps=0, pivotrule=lambda D: bland(D, eps=0), verbose=False):
    d = Dictionary(c, a, b, dtype)
    return simplex(d, eps, pivotrule, verbose)


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
# If LP is infeasible the return value is LPResult.INFEASIBLE, None
# We assume the dictionary is infeasible if the origin solution is feasible.
#
# If LP is unbounded the return value is LPResult.UNBOUNDED, None
# A dictionary on standard form is unbounded if, for any non-zero variable in the OF all restrictions are positive.
#
# If LP has an optimal solution the return value is
# LPResult.OPTIMAL,d, where d is an optimal dictionary.
#
#
# In general when working with fractions one has to divided periodically by the common factors, to prevent an explosion
# of the numerator and the denominator. Note that this is done automatically when using pythons Fraction type.
#
# 0) Set max_degenerate_steps_before_anti_cycle_mode to whatever you choose and
#    consecutive_degenerate_steps = 0
# 1) Check if dictionary is not origo-feasible
#   True: return LPResult.INFEASIBLE, None
# 2) Get entering and leaving variable using the pivot rule
# 3) While entering and leaving is None (We haven't found the optimal, nor that the dictionary is unbounded)
#       a) pivot(entering, leaving)
#       b) Check if dictionary is degenerate (If any of the constraint constants are zero)
#           True: Add one to count of consecutive degenerate steps
#                 Check if consecutive degenerate steps have reached the max_degenerate_steps_before_anti_cycle_mode
#                       True: Shift to an anti cycle mode (We use bland's rule for anti-cycle pivoting)
#           False: Set consecutive degenerate steps = 0
#       c) Get next entering and leaving variable using the pivot rule
# 4) Check if the dictionary is unbounded (entering is not None and leaving is None)
#   True: return LPResult.UNBOUNDED, None
# 5) return LPResult.OPTIMAL, d
def simplex(d, eps=0, pivotrule=lambda d, eps: bland(d, eps=0), verbose=False):
    consecutive_degenerate_steps_before_anti_cycle = 10
    if (d.C[1:, 0] < 0).any():
        return LPResult.INFEASIBLE, None
    consecutive_degenerate_steps = 0
    entering, leaving = pivotrule(d, eps)
    while entering is not None and leaving is not None:
        d.pivot(entering, leaving)
        for const in d.C[:, 0][1:]:
            if eps_correction(const, eps, d.dtype) == 0:  # New dictionary is degenerate
                consecutive_degenerate_steps += 1
                if consecutive_degenerate_steps > consecutive_degenerate_steps_before_anti_cycle:
                    pivotrule = lambda d, eps: bland(d, eps)
                break
        else:
            consecutive_degenerate_steps = 0
        entering, leaving = pivotrule(d, eps)
    if entering is not None and leaving is None:
        return LPResult.UNBOUNDED, None
    return LPResult.OPTIMAL, d


def position_of_auxiliary_variable_in_basis(d: dictionary.Dictionary):
    # The auxiliary variable-name is at varname[n+1] (n+1 = cols-1 in C). Thus we look for this entry in the basis.
    rows, cols = d.C.shape
    for leaving, entry in enumerate(d.B):
        if entry == cols - 1:
            return True, leaving
    return False, None


def basis_index_auxiliary_variable(d: dictionary.Dictionary):
    # The auxiliary variable-name is at varname[n+1] (n+1 = cols-1 in C). Thus we look for this entry in the basis.
    rows, cols = d.C.shape
    for index, entry in enumerate(d.N):
        if entry == cols - 1:
            return index
    return None


def lowest_constraint_const(d):
    cols = d.C.shape[0]
    lowest_const = 0
    basic_variable = 0
    for column in range(1, cols):
        if lowest_const > d.C[column, 0]:
            lowest_const = d.C[column, 0]
            basic_variable = column - 1
    return basic_variable


def linprog(c, a_ub=None, b_ub=None, a_eq=None, b_eq=None):
    res = linprog_original(-c,
                           A_ub=a_ub,
                           b_ub=b_ub,
                           A_eq=a_eq,
                           b_eq=b_eq,
                           method='simplex')  # We need to say that the optimization should use the simplex method
    return res


def dictionary_to_input_arrays(d):
    c = d.C[0, 1:]
    a_ub = -d.C[1:, 1:]
    b_ub = d.C[1:, 0]
    objective_function_constant = d.C[0, 0]
    if objective_function_constant == 0:
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
        return c_plus, a_ub_plus, b_ub, a_eq, b_eq
