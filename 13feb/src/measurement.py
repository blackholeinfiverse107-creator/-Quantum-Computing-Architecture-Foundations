import numpy as np
from typing import List, Tuple, Optional
from src.quantum_state import QuantumState
from src.operators import LinearOperator

class ProjectiveMeasurement:
    """
    Represents a projective measurement defined by a set of projectors {P_m}.
    Enforces completeness relation: sum(P_m) = I.
    """
    def __init__(self, projectors: List[LinearOperator]):
        self._projectors = projectors
        self._validate_completeness()

    def _validate_completeness(self):
        dim = self._projectors[0].dim
        total = np.zeros((dim, dim), dtype=np.complex128)
        for proj in self._projectors:
            if proj.dim != dim:
                 raise ValueError("All projectors must have the same dimension.")
            total += proj.matrix
        
        if not np.allclose(total, np.eye(dim)):
            raise ValueError("Projectors do not sum to Identity (Completeness relation violated).")

    def measure(self, state: QuantumState, seed: Optional[int] = None) -> Tuple[int, QuantumState, float]:
        """
        Performs the measurement on the state.
        
        Args:
            state: The QuantumState to measure.
            seed: Optional seed for deterministic collapse.
            
        Returns:
            Tuple containing:
            - measured_index: Index of the outcome (0, 1, ...)
            - collapsed_state: The new QuantumState after collapse.
            - probability: The probability of this outcome.
        """
        if self._projectors[0].dim != len(state.vector):
            raise ValueError("Measurement dimension does not match state dimension.")

        probabilities = []
        possible_states = []
        
        # Calculate probabilities for each outcome
        for P in self._projectors:
            # P|psi>
            projected_vector = P.matrix @ state.vector
            
            # prob = <psi|P|psi> = ||P|psi>||^2 (since P is Hermitian/Projector)
            # Or strictly <psi|P^dag P|psi> if P is just a projector. 
            # For standard projective measurement, P is Hermitian and P^2=P.
            # prob = <psi|P|psi>
            
            prob = float(np.real(np.vdot(state.vector, projected_vector)))
            
            # Handle numerical noise (probs slightly < 0)
            if prob < 0:
                prob = 0.0
            
            probabilities.append(prob)
            possible_states.append(projected_vector)

        probabilities = np.array(probabilities)
        sum_probs = np.sum(probabilities)
        
        if not np.isclose(sum_probs, 1.0):
             # Renormalize probabilities if they drifted slightly due to float precision
             # provided they are close enough to 1.
             if abs(sum_probs - 1.0) > 1e-5:
                 raise ValueError(f"Probabilities sum to {sum_probs}, expected 1.0.")
             probabilities = probabilities / sum_probs

        # Select outcome
        rng = np.random.default_rng(seed)
        outcome_idx = rng.choice(len(probabilities), p=probabilities)
        
        # Collapse and normalize
        collapsed_vector = possible_states[outcome_idx]
        norm = np.linalg.norm(collapsed_vector)
        
        if np.isclose(norm, 0):
             raise ValueError("Selected outcome has zero probability/norm (should not happen).")
             
        # Create new state (QuantumState class handles normalization)
        new_state = QuantumState(collapsed_vector / norm)
        
        return int(outcome_idx), new_state, probabilities[outcome_idx]

def measure_z_basis() -> ProjectiveMeasurement:
    # P0 = |0><0| = [[1, 0], [0, 0]]
    # P1 = |1><1| = [[0, 0], [0, 1]]
    P0 = LinearOperator([[1, 0], [0, 0]])
    P1 = LinearOperator([[0, 0], [0, 1]])
    return ProjectiveMeasurement([P0, P1])
