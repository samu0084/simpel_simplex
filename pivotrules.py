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
def bland(d, eps, dtype, verbose=False):
    # 1) Initialize entering and leaving to None
    entering = leaving = None
    # 2) Iterate over the objective function variable coefficients
    for col in range(1, d.C.shape[1]):
        # a) Check if eps-corrected coefficient is non-positive
        value = eps_correction(d.C[0, col], eps, dtype)
        if value <= 0:
            # True: continue with next iteration
            continue
        # b) Set entering to the current variable and break out
        entering = col - 1
        break
    # 3) Check if entering is None
    # (If so then all variable coefficients in the OF is non-positive, and thus the dictionary is optimal)
    if entering is None:
        # True: return None, None
        return None, None
    # 4) Sat leaving variable to which limits the growth of the entering variable the most. (Use helper function)
    leaving, _ = leaving_variable(d, eps, entering, dtype)
    # 5) return entering and leaving
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
#   a) Get coefficient value through eps_correction TODO: Check eps_correction and write tests for it
#   b) Check if eps-corrected coefficient value is greater than largest_until_now
#       True: Set largest_until_now to coefficient value
#             Set entering to variable index in N of the current coefficient
# 3) Check if entering is None
#   True: return None, None
# 4) return entering, leaving variable which limits the growth of the entering variable the most
def largest_coefficient(d, eps, dtype, verbose=False):
    entering = leaving = None
    largest_found = 0
    for col in range(1, d.C.shape[1]):
        current_value = eps_correction(d.C[0, col], eps, dtype)
        if largest_found < current_value:
            largest_found = current_value
            entering = col - 1
    if entering is None:  # Is optimal
        return entering, leaving
    leaving, _ = leaving_variable(d, eps, entering, dtype)
    return entering, leaving

# Pick as entering variable, the variable which can be increased the most.
#
# 1) initialize entering and leaving to None and
#    largest_ratio_until_now = -inf (Note that ratio and variable increase is the same thing)
# 2) for each variable in the OF
#   a) Check if eps-corrected variable coefficient is non-positive in OF
#       True: continue with next iteration
#   b) Find potential leaving variable and corresponding ratio, which limits the potential entering variable the most.
#      (Use helper) TODO: Check that the helper functions leaving variable is None if the dictionary is unbounded
#   c) Check if ratio is greater than largest_ratio_until_now
#       True: set largest_ratio_until_now = ratio
#             entering = the variable we are currently considering in the for loop
#             leaving = potential_leaving_variable
# 3) return entering and leaving variable
def largest_increase2(D, eps):
    k = l = None
    temporary_ratios = np.full(len(D.C[1, :]), np.inf)
    index = -1

    # Finds entering varaible by choosing the one that increases the objective function the most
    for i, x in enumerate(D.C[0, 1:]):
        xk = zip(D.C[1:, 0], D.C[1:, i + 1])
        # calulates ratio for each vaar with positive coeffcient in the objective function
        ratios = [abs(x / c) if not ((-eps <= c <= eps) or x >= -eps) else math.inf for c, x in k]
        # Choose the one closet to zero (takes absolute value of each ratio)
        if x > 0 and min(ratios) < min(temporary_rations):
            temporary_rations = ratios
            index = i
    if index == -1:  # No one to pivot
        return None, None
    k = index
    l = leaving_var(D, k, eps)
    return k, l


def leaving_var(D, k, eps):
    xk = zip(D.C[1:, 0], D.C[1:, k + 1])
    ratios = []
    basic_vars = [i for i in D.B]

    for c, x in xk:
        if x >= -eps:
            ratios.append(math.inf)
        elif -eps <= c <= eps:
            ratios.append(100000)
        else:
            ratios.append(x / c)

    ratios = np.array(ratios)
    smallest_ratio = np.where(ratios == np.min(ratios))

    if len(smallest_ratio) == 1:
        l = np.argmin(ratios)
        return l

    # find smallest x from those with same ratio
    cur_smallest = math.inf
    for l_index in smallest_ratio:
        x_name = basic_vars[l_index]
        if x_name < cur_smallest:
            cur_smallest = x_name
    l = cur_smallest

    return l


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
#   c) Calculate how much the OF is increased: OF_coefficient * ratio
#   d) Check if increase is greater than largest_increase_until_now
#       True: largest_increase_until_now = increase
#             entering = current potential entering variable
#             leaving = potential leaving variable
# 3) return found results
def largest_increase(d, eps, verbose=False):
    entering = leaving = None
    best_until_now = 0

    for col in range(1, d.C.shape[1]):
        if verbose:
            print(f"Looking into col looks like this: {d.C[:, col]}")
        if eps_correction(d.C[0, col], eps) <= 0:
            continue
        leaving_given_col, ratio = leaving_variable(d, eps, col - 1, verbose)
        if verbose:
            print(
                f"maybe leaving = {leaving_given_col}; ratio = {ratio}")  # TODO Can it happen that ratio is not assigned on first check? Chich this in the method "leaving variable"
        if leaving_given_col is None:
            entering = col - 1  # In case of unbounded we need to return Some, none
            break
        if verbose:
            print(f"increase = d.C[0, col] * ratio: {d.C[0, col] * ratio} = {d.C[0, col]} * {ratio}")
        increase = d.C[0, col] * ratio
        if verbose:
            print(f"increase = {increase}")
            print(f"best_until_now < increase     {best_until_now} < {increase}")
        if best_until_now < increase:
            best_until_now = increase
            entering = col - 1
            leaving = leaving_given_col
            if verbose:
                print(f"best_until_now = increase : {increase}")
                print(f"entering = col - 1 : {col - 1}")
                print(f"leaving = leaving_given_col : {leaving_given_col}")
    return entering, leaving


# Pick leaving variable which limits the growth of the entering variable the most.
# 1) Initialize the variable least_until_now to be infinite and
#    the leaving variable to be None
# 2) for each constraint:
#   a) Check if constraint coefficient of entering variable is non-negative
#      (Note: For a tableau or algebraic notation the coefficient would have to be positive instead)
#       True: go to next iteration
#   b) Calculate: ratio = constraint constant / negative constraint coefficient of entering variable
#   c) Check if ratio is less than least until now
#       True: set leas_until_now = ratio
#             Save leaving variable
def leaving_variable(d, eps, entering, dtype, verbose=False):
    leaving = None
    smallest_ratio = 0
    for row in range(1, d.C.shape[0]):
        coefficient = eps_correction(d.C[row, entering + 1], eps, dtype)
        if coefficient >= 0:
            continue
        constant = eps_correction(d.C[row, 0], eps, dtype)
        # if constant == 0:  # TODO: Is this correct?
        #     continue
        new_ratio = constant / -coefficient
        if leaving is None:
            smallest_ratio = new_ratio
            leaving = row - 1
        if smallest_ratio > new_ratio:
            smallest_ratio = new_ratio
            leaving = row - 1
    return leaving, smallest_ratio


def eps_correction(value, eps, dtype):
    # eps>=0 is such that numbers in the closed interval [-eps,eps]
    # are to be treated as if they were 0
    if dtype == np.float64 or eps <= 0:
        return value
    if -eps <= value <= eps:
        return 0
    else:
        return value
