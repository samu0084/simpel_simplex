def primal_to_dual(c, a, b):
    return b * -1, a.Transpose * -1, c * -1
