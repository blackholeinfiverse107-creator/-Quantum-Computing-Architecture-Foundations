import numpy as np
from src.operators import UnitaryOperator

# 1-Qubit Gates
def I() -> UnitaryOperator:
    return UnitaryOperator([[1, 0], [0, 1]])

def X() -> UnitaryOperator:
    return UnitaryOperator([[0, 1], [1, 0]])

def Y() -> UnitaryOperator:
    return UnitaryOperator([[0, -1j], [1j, 0]])

def Z() -> UnitaryOperator:
    return UnitaryOperator([[1, 0], [0, -1]])

def H() -> UnitaryOperator:
    val = 1 / np.sqrt(2)
    return UnitaryOperator([[val, val], [val, -val]])

# 2-Qubit Gates
def CNOT() -> UnitaryOperator:
    # Control on qubit 0, Target on qubit 1 (standard convention |c t>)
    return UnitaryOperator([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ])
