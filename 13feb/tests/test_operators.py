import unittest
import numpy as np
from src.quantum_state import QuantumState
from src.operators import LinearOperator, UnitaryOperator
from src.gates import X, H, Z, CNOT, I

class TestOperators(unittest.TestCase):
    def test_linear_operator_init(self):
        mat = [[1, 0], [0, 1]]
        op = LinearOperator(mat)
        np.testing.assert_array_equal(op.matrix, np.array(mat))

    def test_unitary_validation(self):
        # Valid unitary (X)
        try:
            UnitaryOperator([[0, 1], [1, 0]])
        except ValueError:
            self.fail("UnitaryOperator raised ValueError for valid unitary X")

        # Invalid unitary
        with self.assertRaises(ValueError):
            UnitaryOperator([[1, 1], [1, 1]])

    def test_gate_application(self):
        # H|0> -> |+>
        q0 = QuantumState([1, 0])
        h_op = H()
        q_plus = h_op.apply(q0)
        
        expected = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
        np.testing.assert_array_almost_equal(q_plus.vector, expected)

    def test_pauli_x(self):
        # X|0> -> |1>
        q0 = QuantumState([1, 0])
        q1 = X().apply(q0)
        np.testing.assert_array_almost_equal(q1.vector, np.array([0, 1]))

    def test_cnot(self):
        # CNOT |10> -> |11> (Control is qubit 0)
        q1 = QuantumState([0, 1])
        q0 = QuantumState([1, 0])
        
        # State |1> (x) |0> = |10> = [0, 0, 1, 0]
        q_10 = q1.tensor_product(q0) 
        
        # CNOT |10> -> |11>
        q_final = CNOT().apply(q_10)
        
        # |11> = [0, 0, 0, 1]
        np.testing.assert_array_almost_equal(q_final.vector, np.array([0, 0, 0, 1]))

    def test_tensor_product_operators(self):
        # (X (x) I) |00> -> |10>
        # applying X to first qubit, I to second
        op = X().tensor_product(I())
        state = QuantumState([1, 0]).tensor_product(QuantumState([1, 0])) # |00>
        
        new_state = op.apply(state)
        # |10>
        expected = np.array([0, 0, 1, 0]) 
        np.testing.assert_array_almost_equal(new_state.vector, expected)

if __name__ == '__main__':
    unittest.main()
