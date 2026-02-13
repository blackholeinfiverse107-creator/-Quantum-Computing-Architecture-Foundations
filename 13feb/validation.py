import numpy as np
from src.quantum_state import QuantumState
from src.gates import H, CNOT, X
from src.measurement import measure_z_basis

def match_state(state, target_vector):
    return np.allclose(state.vector, target_vector)

def run_validation():
    print("=== Validation: Formal Quantum State Model ===")
    
    # 1. Superposition
    print("\n[Test 1] creating Superposition...")
    q0 = QuantumState([1, 0])
    q_plus = H().apply(q0)
    print(f"Norm after H: {q_plus.norm():.5f}")
    if match_state(q_plus, np.array([1/np.sqrt(2), 1/np.sqrt(2)])):
        print("PASS: |+> state created correctly.")
    else:
        print("FAIL: Superposition state incorrect.")

    # 2. Entanglement (Bell State)
    print("\n[Test 2] Creating Bell State (|00> + |11>)/sqrt(2)...")
    q0 = QuantumState([1, 0])
    q1 = QuantumState([1, 0])
    q00 = q0.tensor_product(q1) # |00>
    
    # Apply H to first qubit -> (|0>+|1>)|0> = |00> + |10>
    # Note: Tensor product order matters. 
    # If we treat q0 as qubit 0 (control) and q1 as qubit 1 (target).
    # We need an operator H (x) I.
    from src.gates import I
    op_h_i = H().tensor_product(I())
    psi_step1 = op_h_i.apply(q00)
    
    # Apply CNOT (Control 0, Target 1)
    psi_bell = CNOT().apply(psi_step1)
    
    expected_bell = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
    if match_state(psi_bell, expected_bell):
        print("PASS: Bell state created correctly.")
    else:
        print("FAIL: Bell state creation failed.")
        print(f"State: {psi_bell.vector}")

    # 3. Measurement Collapse
    print("\n[Test 3] Measuring Bell State...")
    meas = measure_z_basis()
    
    # We can't easily measure just one qubit with our current `ProjectiveMeasurement` 
    # unless we define projectors for the full 4D space.
    # Logic: Measure qubit 0. P0_full = |0><0| (x) I, P1_full = |1><1| (x) I.
    
    from src.operators import LinearOperator
    P0 = LinearOperator([[1, 0], [0, 0]])
    P1 = LinearOperator([[0, 0], [0, 1]])
    I_op = I()
    
    P0_full = P0.tensor_product(I_op)
    P1_full = P1.tensor_product(I_op)
    
    meas_q0 = ProjectiveMeasurement([P0_full, P1_full]) # Measure 1st qubit
    
    # Seed for determinism needed? Let's try to verify correlation.
    outcome, collapsed_state, prob = meas_q0.measure(psi_bell)
    
    print(f"Measured Qubit 0: {outcome} (Prob: {prob:.2f})")
    
    # If 0, state should be |00>
    # If 1, state should be |11>
    if outcome == 0:
        expected = np.array([1, 0, 0, 0])
    else:
        expected = np.array([0, 0, 0, 1])
        
    if match_state(collapsed_state, expected):
        print("PASS: Collapse to correct product state.")
    else:
        print("FAIL: Incorrect collapse.")
        print(f"Expected: {expected}")
        print(f"Got: {collapsed_state.vector}")
        
    # 4. No-Cloning / Immutability
    print("\n[Test 4] Verifying Immutability...")
    vec = psi_bell.vector
    vec[0] = 999 
    if psi_bell.vector[0] != 999:
        print("PASS: State vector is immutable (copy returned).")
    else:
        print("FAIL: State vector was mutated!")

if __name__ == "__main__":
    run_validation()
