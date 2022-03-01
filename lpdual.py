def primal_to_dual(c, a, b):
    return b * -1, a.transpose() * -1, c * -1
