import unittest
import numpy as np
from src.quantum_state import QuantumState

class TestQuantumState(unittest.TestCase):
    def test_initialization_normalization(self):
        # |0> state
        state = QuantumState([1, 0])
        self.assertAlmostEqual(state.norm(), 1.0)
        np.testing.assert_array_almost_equal(state.vector, np.array([1, 0], dtype=np.complex128))

        # Unnormalized state
        state_unnorm = QuantumState([1, 1])
        self.assertAlmostEqual(state_unnorm.norm(), 1.0)
        expected = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=np.complex128)
        np.testing.assert_array_almost_equal(state_unnorm.vector, expected)

    def test_zero_vector_raises(self):
        with self.assertRaises(ValueError):
            QuantumState([0, 0])

    def test_invalid_dimension(self):
        with self.assertRaises(ValueError):
            QuantumState([1, 0, 0]) # dim 3

    def test_tensor_product(self):
        q0 = QuantumState([1, 0]) # |0>
        q1 = QuantumState([0, 1]) # |1>
        
        # |0> (x) |1> = |01> = [0, 1, 0, 0]
        combined = q0.tensor_product(q1)
        expected = np.array([0, 1, 0, 0], dtype=np.complex128)
        
        self.assertEqual(combined.num_qubits, 2)
        np.testing.assert_array_almost_equal(combined.vector, expected)

    def test_immutability(self):
        state = QuantumState([1, 0])
        vec = state.vector
        vec[0] = 0 # Modify copy
        self.assertEqual(state.vector[0], 1.0+0j) # Original unchanged

if __name__ == '__main__':
    unittest.main()
