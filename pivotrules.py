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
    leaving, ratio = leaving_variable(d, eps, entering)
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
    for col in range(1, d.C.shape[1]):
        current_value = eps_correction(d.C[0, col], eps)
        if largest_found < current_value:
            largest_found = current_value
            entering = col - 1
    if entering is None:  # Is optimal
        return entering, leaving
    # pick leaving variable with greatest ratio:
    # constrin_const / constrin_coefficient_of_leaving_variable
    leaving, ratio = leaving_variable(d, eps, entering)
    return entering, leaving


def largest_increase(d, eps, verbose=False):
    # Assumes a feasible dictionary D and find entering and leaving
    # variables according to the Largest Increase rule.
    #
    # eps>=0 is such that numbers in the closed interval [-eps,eps]
    # are to be treated as if they were 0
    #
    # TODO Returns entering and leaving such that
    # entering is None if D is Optimal
    # Otherwise D.N[entering] is entering variable
    # leaving is None if D is Unbounded
    # Otherwise D.B[leaving] is a leaving variable

    entering = leaving = None
    best_until_now = 0

    for col in range(1, d.C.shape[1]):
        if verbose:
            print(f"Looking into col looks like this: {d.C[:, col]}")
        if eps_correction(d.C[0, col], eps) <= 0:
            continue
        leaving_given_col, ratio = leaving_variable(d, eps, col - 1, verbose)
        if verbose:
            print(f"maybe leaving = {leaving_given_col}; ratio = {ratio}") # TODO Can it happen that ratio is not assigned on first check? Chich this in the method "leaving variable"
        if leaving_given_col is None:
            continue
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


def leaving_variable(d, eps, entering, verbose=False):
    # Pick leaving variable with the smallest numerical ratio:
    # constrin_const / constrin_coefficient_of_leaving_variable
    leaving = None
    smallest_ratio = 0
    for row in range(1, d.C.shape[0]):
        if verbose:
            print(f"row = {row}")
            print(f"entering + 1 = {entering + 1}")
            print(f"coefficient = {eps_correction(d.C[row, entering + 1], eps)}")
        coefficient = eps_correction(d.C[row, entering + 1], eps)
        if coefficient >= 0:
            continue
        constant = eps_correction(d.C[row, 0], eps)
        new_ratio = constant / -coefficient
        if verbose:
            print(f"constant = {eps_correction(d.C[row, 0], eps)}")
            print(f"new_ratio = {constant / coefficient}")
        if leaving is None:
            smallest_ratio = new_ratio
            leaving = row - 1
            if verbose:
                print(f"new smallest_ratio =  {smallest_ratio}")
                print(f"new leaving = {leaving}")
                print()
        if smallest_ratio > new_ratio:
            smallest_ratio = new_ratio
            leaving = row - 1
            if verbose:
                print(f"new smallest_ratio =  {smallest_ratio}")
                print(f"new leaving = {leaving}")
                print()
    print()
    return leaving, smallest_ratio


def eps_correction(value, eps):
    # eps>=0 is such that numbers in the closed interval [-eps,eps]
    # are to be treated as if they were 0
    if eps < 0:
        return value
    if -eps <= value <= eps:
        return 0
    else:
        return value
