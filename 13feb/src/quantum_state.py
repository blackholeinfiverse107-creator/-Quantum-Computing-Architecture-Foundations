import numpy as np
from typing import Union, List, Tuple

class QuantumState:
    """
    Represents a pure quantum state vector in a Hilbert space.
    Enforces normalization and immutability.
    """
    def __init__(self, vector: Union[List[complex], np.ndarray]):
        """
        Initialize a QuantumState from a vector.
        The vector is automatically normalized.
        
        Args:
            vector: A list of complex numbers or a numpy array representing the state vector.
            
        Raises:
            ValueError: If the vector norm is 0 or if the dimension is not a power of 2.
        """
        self._vector = np.array(vector, dtype=np.complex128).flatten()
        norm = np.linalg.norm(self._vector)
        
        if np.isclose(norm, 0):
            raise ValueError("State vector cannot be zero vector.")
            
        # Enforce normalization Strict
        self._vector = self._vector / norm
        self._set_dimension()

    def _set_dimension(self):
        """Validates dimension is a power of 2 (qubits)."""
        dim = len(self._vector)
        if dim == 0 or (dim & (dim - 1) != 0):
             raise ValueError(f"Dimension {dim} is not valid for a qubit system (must be power of 2).")
        self._dim = dim
        self._num_qubits = int(np.log2(dim))

    @property
    def vector(self) -> np.ndarray:
        """Returns a copy of the state vector to ensure immutability."""
        return self._vector.copy()
    
    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    def norm(self) -> float:
        """Returns the L2 norm of the state (should always be approx 1.0)."""
        return float(np.linalg.norm(self._vector))

    def tensor_product(self, other: 'QuantumState') -> 'QuantumState':
        """
        Computes the tensor product of this state with another state.
        result = |self> (x) |other>
        """
        new_vector = np.kron(self._vector, other._vector)
        return QuantumState(new_vector)

    def __repr__(self):
        return f"QuantumState(dim={self._dim}, qubits={self._num_qubits})"

    def __eq__(self, other):
        if not isinstance(other, QuantumState):
            return False
        # Phase agnostic equality could be implemented, but strict equality is safer for now
        # allowing for global phase factor might be needed later, but strict vector equality for now.
        if self._dim != other._dim:
            return False
        return np.allclose(self._vector, other._vector)
