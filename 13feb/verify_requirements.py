import numpy as np
from src.quantum_state import QuantumState
from src.operators import LinearOperator, UnitaryOperator
from src.measurement import measure_z_basis
from src.gates import H, X, Z, I, CNOT

def log_result(req_name, passed, details=""):
    status = "NON-COMPLIANT"
    if passed:
        status = "COMPLIANT"
    print(f"[{status}] {req_name}: {details}")

def verify_requirements():
    print("=== Formal Quantum Model Requirement Verification ===\n")

    # 1. State Vectors & Normalization
    try:
        # Test: Auto-normalization
        v = [1, 1]
        qs = QuantumState(v)
        norm = qs.norm()
        # Expected: 1/sqrt(2) * [1, 1] -> norm 1.0
        passed = np.isclose(norm, 1.0) and np.allclose(qs.vector, np.array(v)/np.linalg.norm(v))
        log_result("State Representation & Normalization", passed, f"Norm={norm}")
    except Exception as e:
        log_result("State Representation & Normalization", False, str(e))

    # 2. Linearity of Operators
    try:
        # Test: A(a|u> + b|v>) == aA|u> + bA|v>
        u_vec = [1, 0]
        v_vec = [0, 1]
        alpha = 0.6
        beta = 0.8
        
        # We manually construct input vectors because QuantumState enforces normalization on init, 
        # making arbitrary linear combinations tricky to represent as "One State" if they aren't normalized.
        # But Linearity applies to the Operator class logic.
        
        op = H()
        
        # LHS: Apply to linear combination
        combined_vec = alpha * np.array(u_vec) + beta * np.array(v_vec)
        # Note: combined_vec norm is sqrt(0.36 + 0.64) = 1.0, so valid state
        state_combined = QuantumState(combined_vec)
        lhs = op.apply(state_combined).vector
        
        # RHS: Linear combination of outputs
        rhs_u = alpha * op.apply(QuantumState(u_vec)).vector
        rhs_v = beta * op.apply(QuantumState(v_vec)).vector
        rhs = rhs_u + rhs_v
        
        passed = np.allclose(lhs, rhs)
        log_result("Linearity Constraint", passed, "Superposition preserved under evolution")
    except Exception as e:
        log_result("Linearity Constraint", False, str(e))

    # 3. Unitary Evolution & Norm Preservation
    try:
        q = QuantumState([1, 0])
        U = H()
        q_new = U.apply(q)
        passed = np.isclose(q_new.norm(), 1.0)
        
        # Test Non-Unitary rejection
        try:
            # Matrix with eigenvalue > 1
            BadOp = UnitaryOperator([[2, 0], [0, 2]]) 
            passed_rejection = False
        except ValueError:
            passed_rejection = True
            
        log_result("Unitary Evolution & Norm Preservation", passed and passed_rejection, "Norm=1 preserved, Invalid U rejected")
    except Exception as e:
        log_result("Unitary Evolution & Norm Preservation", False, str(e))

    # 4. Measurement & Collapse
    try:
        # Bell state: (|00> + |11>)/sqrt(2)
        q = CNOT().apply(H().apply(QuantumState([1, 0])).tensor_product(QuantumState([1, 0])))
        
        meas = measure_z_basis() # This measures a single qubit linespace if 2D, need 4D for Bell?
        
        # Our measure_z_basis is 2D. We need to measure one qubit of the Bell pair.
        # Construct P0 x I and P1 x I
        P0 = LinearOperator([[1, 0], [0, 0]])
        P1 = LinearOperator([[0, 0], [0, 1]])
        P0_I = P0.tensor_product(I())
        P1_I = P1.tensor_product(I())
        
        from src.measurement import ProjectiveMeasurement
        meas_bell = ProjectiveMeasurement([P0_I, P1_I])
        
        # Force a measurement
        outcome, collapsed_state, prob = meas_bell.measure(q, seed=42)
        
        # Check collapse
        if outcome == 0:
             # Should be |00>
             expected = np.array([1, 0, 0, 0])
        else:
             expected = np.array([0, 0, 0, 1])
             
        is_collapsed = np.allclose(collapsed_state.vector, expected)
        # Check post-measurement norm
        norm_ok = np.isclose(collapsed_state.norm(), 1.0)
        
        log_result("Measurement Collapse", is_collapsed and norm_ok, f"Outcome={outcome}, Collapsed Correctly")
        
    except Exception as e:
        log_result("Measurement Collapse", False, str(e))

    # 5. Immutability / Hidden Mutation
    try:
        q = QuantumState([1, 0])
        original_vec = q.vector
        
        # Attempt to modify returned vector
        v = q.vector
        v[0] = 0
        
        # Attempt to modify via operator
        q2 = X().apply(q)
        
        # Verify original q is untouched
        passed = np.allclose(q.vector, [1, 0]) and np.allclose(original_vec, [1, 0])
        log_result("No Hidden Mutation", passed, "Original state remains invariant")
    except Exception as e:
        log_result("No Hidden Mutation", False, str(e))

    # 6. Tensor Product Structure
    try:
        q1 = QuantumState([1, 0])
        q2 = QuantumState([0, 1])
        q12 = q1.tensor_product(q2)
        
        passed_dim = (q12.num_qubits == 2)
        passed_vec = np.allclose(q12.vector, [0, 1, 0, 0])
        
        log_result("Tensor Product Structure", passed_dim and passed_vec, "Correctly formed 2-qubit state")
    except Exception as e:
        log_result("Tensor Product Structure", False, str(e))

if __name__ == "__main__":
    verify_requirements()
