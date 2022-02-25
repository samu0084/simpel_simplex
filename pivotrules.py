def bland(d, eps, verbose=False):
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
    entering = leaving = None
    for col in range(1, d.C.shape[1]):
        value = eps_correction(d.C[0, col], eps)
        if value > 0:
            entering = col - 1
            break
    if entering is None:  # Is optimal
        return entering, leaving
    leaving = leaving_variable(d, eps, entering)
    return entering, leaving


def largest_coefficient(d, eps, verbose=False):
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

    # pick entering variable with largest coefficient in z
    entering = leaving = None
    largest_found = 0
    for index in range(1, d.C.shape[1]):
        current_value = eps_correction(d.C[0, index], eps)
        if largest_found < current_value:
            largest_found = current_value
            entering = index - 1
    if entering is None:  # Is optimal
        return entering, leaving
    # pick leaving variable with greatest ratio:
    # constrin_const / constrin_coefficient_of_leaving_variable
    leaving = leaving_variable(d, eps, entering)
    return entering, leaving


def largest_increase(d, eps, verbose=False):
    # Assumes a feasible dictionary D and find entering and leaving
    # variables according to the Largest Increase rule.
    #
    # eps>=0 is such that numbers in the closed interval [-eps,eps]
    # are to be treated as if they were 0
    #
    # Returns k and l such that
    # k is None if D is Optimal
    # Otherwise D.N[k] is entering variable
    # l is None if D is Unbounded
    # Otherwise D.B[l] is a leaving variable

    k = l = None
    # TODO
    return k, l


def leaving_variable(d, eps, k, verbose=False):
    # pick leaving variable with greatest ratio:
    # constrin_const / constrin_coefficient_of_leaving_variable
    l = None
    for index in range(1, d.C.shape[0]):
        if verbose:
            print(f"constraint_coefficient = {eps_correction(d.C[index, k + 1], eps)}")
        constraint_coefficient = eps_correction(d.C[index, k + 1], eps)
        if constraint_coefficient >= 0:
            continue
        constraint_constant = eps_correction(d.C[index, 0], eps)
        new_ratio = constraint_constant / constraint_coefficient
        if verbose:
            print(f"constraint_constant = {eps_correction(d.C[index, 0], eps)}")
            print(f"new_ratio = {constraint_constant / constraint_coefficient}")
        if l == None:
            largest_ratio = new_ratio
            l = index - 1
            if verbose:
                print(f"new largest_ratio =  {largest_ratio}")
                print(f"new l = {l}")
                print()
        if largest_ratio < new_ratio:
            largest_ratio = new_ratio
            l = index - 1
            if verbose:
                print(f"new largest_ratio =  {largest_ratio}")
                print(f"new l = {l}")
                print()

    return l


def eps_correction(value, eps):
    # eps>=0 is such that numbers in the closed interval [-eps,eps]
    # are to be treated as if they were 0
    if eps < 0:
        return value
    if -eps <= value <= eps:
        return 0
    else:
        return value
