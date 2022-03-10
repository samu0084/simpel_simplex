import math

import numpy as np


# Assumes a feasible dictionary D and finds entering and leaving
# variables according to Bland's rule.
#
# eps>=0 is such that numbers in the closed interval [-eps,eps]
# are to be treated as if they were 0
#
# Returns entering and leaving such that
# entering is None if D is Optimal
# Otherwise D.N[entering] is entering variable
# leaving is None if D is Unbounded
# Otherwise D.B[leaving] is a leaving variable
#
# Pick the leftmost variable in the objective function, for which the coefficient is positive, as entering variable
# Pick the variable from the basis for which the corresponding constraint limits the growth of the entering variable
# the most as leaving variable
#
# 1) Initialize entering and leaving to None
# 2) Iterate over the objective function variable coefficients
#   a) Check if eps-corrected coefficient is non-positive
#       True: continue with next iteration
#   b) Set entering to the current variable
# 3) Check if entering is None
#       True: return None, None
# 4) Pick leaving variable for which the corresponding constraint limits the growth of the entering variable the most.
#    (Use helper function)
# 5) return entering and leaving
def bland(d, eps, verbose=False):
    entering = None
    for col in range(1, d.C.shape[1]):
        value = eps_correction(d.C[0, col], eps, d.dtype)
        if value <= 0:
            continue
        entering = col - 1
        break
    if entering is None:
        return None, None
    leaving, _ = leaving_variable(d, eps, entering)
    return entering, leaving


# Assumes a feasible dictionary D and find entering and leaving
# variables according to the Largest Coefficient rule.
#
# eps>=0 is such that numbers in the closed interval [-eps,eps]
# are to be treated as if they were 0
#
# Returns entering and leaving such that
# entering is None if D is Optimal
# Otherwise D.N[entering] is entering variable
# leaving is None if D is Unbounded
# Otherwise D.B[leaving] is a leaving variable
#
# Pick entering variable with the largest coefficient in z, and
# leaving variable from the row which limits the growth of the entering variable the most
# (Use helper function for that last part)
#
# 1) Initialize entering and leaving to None and
#    largest_until_now to 0
# 2) For each coefficient in OF
#   a) Get coefficient value through eps_correction
#   b) Check if eps-corrected coefficient value is greater than largest_until_now
#       True: Set largest_until_now to coefficient value
#             Set entering to variable index in N of the current coefficient
# 3) Check if entering is None
#   True: return None, None
# 4) return entering, leaving variable which limits the growth of the entering variable the most
def largest_coefficient(d, eps, verbose=False):
    entering = None
    largest_until_now = 0
    for col in range(1, d.C.shape[1]):
        current_value = eps_correction(d.C[0, col], eps, d.dtype)
        if largest_until_now < current_value:
            largest_until_now = current_value
            entering = col - 1
    if entering is None:
        return None, None
    leaving, _ = leaving_variable(d, eps, entering)
    return entering, leaving


# Assumes a feasible dictionary D and find entering and leaving
# variables according to the Largest Increase rule.
#
# eps>=0 is such that numbers in the closed interval [-eps,eps]
# are to be treated as if they were 0
#
# entering is None if D is Optimal
# Otherwise D.N[entering] is entering variable
# leaving is None if D is Unbounded
# Otherwise D.B[leaving] is a leaving variable
#
# Finds the entering and leaving variable which increases the value of the objective function the most
#
# 1) Initialize largest_increase_until_now = 0 and
#    entering and leaving to be None
# 2) For each variable coefficient in the OF
#   a) Check if eps-corrected coefficient value is non-positive
#       True: continue with next iteration.
#   b) Get ratio for the current potential entering variable. (Note that ratio equals the increase of the variable)
#   c) Check if the entering variable is unbounded (If all constraint coefficients a non-basic variable is non-negative
#      then the dictionary is unbounded, and then the 'leaving_variable()' helper method will return None, math.inf)
#       True: In case of unbounded we need to return Some, none, and so we set the entering variable to infinity
#             and the leaving variable to None
#   d) Calculate how much the OF is increased: OF_coefficient * ratio
#   e) Check if increase is greater than largest_increase_until_now
#       True: largest_increase_until_now = increase
#             entering = current potential entering variable
#             leaving = potential leaving variable
# 3) return found results
def largest_increase(d, eps, verbose=False):
    entering = leaving = None
    largest_increase_until_now = -math.inf
    for col in range(1, d.C.shape[1]):
        coefficient = eps_correction(d.C[0, col], eps, d.dtype)
        if coefficient <= 0:
            continue
        leaving_given_col, ratio = leaving_variable(d, eps, col - 1, verbose)
        if leaving_given_col is None:
            entering = math.inf
            leaving = None
            break
        increase = d.C[0, col] * ratio
        if largest_increase_until_now < increase:
            largest_increase_until_now = increase
            entering = col - 1
            leaving = leaving_given_col
    return entering, leaving


# Pick leaving variable which limits the growth of the entering variable the most.
# 1) Initialize the variable least_until_now to be infinite and the leaving variable to be None
# 2) for each constraint:
#   a) Check if constraint coefficient of entering variable is non-negative
#      (Note: For a tableau or algebraic notation the coefficient would have to be positive instead)
#       True: go to next iteration
#   b) Calculate: ratio = constraint constant / negative constraint coefficient of entering variable
#   c) Check if ratio is less than least until now
#       True: set leas_until_now = ratio
#             Save leaving variable
# 3) Check if leaving is None (Implies that all constrain coefficients are positive; thus the problem is unbounded)
#       True: adjust least_until_now to -math.inf to represent unbounded
# 4) return leaving variable, along with least(ratio)-until-now:
def leaving_variable(d, eps, entering, verbose=False):
    leaving = None
    least_until_now = math.inf
    for row in range(1, d.C.shape[0]):
        coefficient = eps_correction(d.C[row, entering + 1], eps, d.dtype)
        if coefficient >= 0:
            continue
        constant = eps_correction(d.C[row, 0], eps, d.dtype)
        new_ratio = constant / -coefficient
        if least_until_now > new_ratio:
            least_until_now = new_ratio
            leaving = row - 1
    if leaving is None:
        least_until_now = -math.inf
    return leaving, least_until_now


# eps>=0 is such that float64 numbers in the closed interval [-eps,eps] are to be treated as if they were 0.
def eps_correction(value, eps, dtype):
    if dtype == np.float64 or eps <= 0:
        return value
    if -eps <= value <= eps:
        return 0
    else:
        return value
