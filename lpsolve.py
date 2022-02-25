from fractions import Fraction

import numpy as np

from dictionary import Dictionary
from lpresult import LPResult
from pivotrules import bland, eps_correction


def lp_solve(c, A, b, dtype=Fraction, eps=0, pivotrule=lambda D: bland(D, eps=0), verbose=False):
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

    return lp_solve_one_phase(c, A, b, dtype, eps, pivotrule, verbose)

    # lp_solve_two_phase(c, A, b, dtype, eps, pivotrule, verbose)


def lp_solve_one_phase(c, A, b, dtype=Fraction, eps=0, pivotrule=lambda D: bland(D, eps=0), verbose=False):
    degenerate_steps_before_anti_cycle = 10
    d = Dictionary(c, A, b, dtype)
    if not (d.basic_solution() == np.zeros(d.N.shape[0])).all():  # infeasible (Without auxiliary)
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
            print(f"Compare with {np.zeros(d.C.shape[0]-1)}")
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
        return LPResult.UNBOUNDED, None
    return LPResult.OPTIMAL, d


def lp_solve_two_phase(c, A, b, dtype=Fraction, eps=0, pivotrule=lambda D: bland(D, eps=0), verbose=False):
    # TODO
    raise NotImplementedError
