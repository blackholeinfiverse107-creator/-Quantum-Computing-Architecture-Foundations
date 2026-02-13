import numpy as np
from typing import Union, List, Optional
from src.quantum_state import QuantumState

class LinearOperator:
    """
    Represents a linear operator acting on a quantum state.
    """
    def __init__(self, matrix: Union[List[List[complex]], np.ndarray]):
        self._matrix = np.array(matrix, dtype=np.complex128)
        if self._matrix.ndim != 2 or self._matrix.shape[0] != self._matrix.shape[1]:
            raise ValueError("Operator must be a square matrix.")
        
        dim = self._matrix.shape[0]
        if dim == 0 or (dim & (dim - 1) != 0):
             raise ValueError(f"Dimension {dim} must be a power of 2.")
        self._dim = dim

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix.copy()
    
    @property
    def dim(self) -> int:
        return self._dim

    def apply(self, state: QuantumState) -> QuantumState:
        """
        Applies the operator to a quantum state: |psi'> = A |psi>
        """
        if self._dim != len(state.vector):
            raise ValueError(f"Operator dimension {self._dim} does not match state dimension {len(state.vector)}.")
        
        new_vector = self._matrix @ state.vector
        # Note: Result (LinearOperator) might not be normalized (unless Unitary).
        # We allow non-unitary operators for general linear algebra, 
        # but QuantumState enforces normalization. 
        # So strictly speaking, A|psi> might not be a valid "QuantumState" if A is not unitary/trace-preserving.
        # However, for the purpose of this framework, we might want to return a raw vector or 
        # re-normalize if it's a valid operation. 
        # BUT, the requirements say "Forbid hidden mutation".
        # If we return a QuantumState, it auto-normalizes. 
        # Let's check if the resulting vector is zero.
        
        if np.allclose(new_vector, 0):
             raise ValueError("Operator mapped state to zero vector (not a valid quantum state).")
             
        return QuantumState(new_vector)
    
    def tensor_product(self, other: 'LinearOperator') -> 'LinearOperator':
        return LinearOperator(np.kron(self._matrix, other._matrix))

    def __matmul__(self, other: 'LinearOperator') -> 'LinearOperator':
        if self._dim != other._dim:
            raise ValueError("Dimension mismatch for matrix multiplication.")
        return LinearOperator(self._matrix @ other._matrix)


class UnitaryOperator(LinearOperator):
    """
    Represents a unitary operator (evolution).
    Enforces U^dagger U = I.
    """
    def __init__(self, matrix: Union[List[List[complex]], np.ndarray], check_unitary: bool = True):
        super().__init__(matrix)
        if check_unitary:
            self._validate_unitary()

    def _validate_unitary(self):
        dagger = self._matrix.conj().T
        product = dagger @ self._matrix
        identity = np.eye(self._dim, dtype=np.complex128)
        if not np.allclose(product, identity):
            raise ValueError("Operator is not Unitary.")

    def apply(self, state: QuantumState) -> QuantumState:
        # Unitary operators preserve norm, so the result is automatically a valid QuantumState
        # without needing aggressive renormalization (though QuantumState init does it anyway).
        return super().apply(state)
        
    def tensor_product(self, other: 'UnitaryOperator') -> 'UnitaryOperator':
        # Tensor product of unitaries is unitary
        new_op = super().tensor_product(other)
        return UnitaryOperator(new_op.matrix, check_unitary=False) # Skip check for efficiency if trusted

    def __matmul__(self, other: 'UnitaryOperator') -> 'UnitaryOperator':
        # Product of unitaries is unitary
        new_op = super().__matmul__(other)
        return UnitaryOperator(new_op.matrix, check_unitary=False)
