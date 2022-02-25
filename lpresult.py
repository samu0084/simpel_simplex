from enum import Enum


class LPResult(Enum):
    OPTIMAL = 1
    INFEASIBLE = 2
    UNBOUNDED = 3
