import unittest
import numpy as np
from src.quantum_state import QuantumState
from src.measurement import ProjectiveMeasurement, measure_z_basis
from src.gates import H

class TestMeasurement(unittest.TestCase):
    def test_z_basis_measure_0(self):
        # Measure |0> -> Should get 0 with prob 1
        q0 = QuantumState([1, 0])
        meas = measure_z_basis()
        outcome, new_state, prob = meas.measure(q0)
        
        self.assertEqual(outcome, 0)
        self.assertAlmostEqual(prob, 1.0)
        np.testing.assert_array_equal(new_state.vector, np.array([1, 0], dtype=np.complex128))

    def test_superposition_collapse(self):
        # Measure |+> = (|0> + |1>)/sqrt(2)
        # Should get 0 or 1 with prob 0.5
        q0 = QuantumState([1, 0])
        q_plus = H().apply(q0)
        
        meas = measure_z_basis()
        
        # Seeded for determinism (force 0)
        # Note: RNG behavior depends on numpy version, but keeping seed constant should strict it.
        # Let's try seed=42
        
        outcome, new_state, prob = meas.measure(q_plus, seed=42)
        
        self.assertIn(outcome, [0, 1])
        self.assertAlmostEqual(prob, 0.5)
        
        # Check collapse
        if outcome == 0:
            expected = np.array([1, 0])
        else:
            expected = np.array([0, 1])
            
        np.testing.assert_array_almost_equal(new_state.vector, expected)

    def test_completeness_check(self):
        # Incomplete projectors
        from src.operators import LinearOperator
        P0 = LinearOperator([[1, 0], [0, 0]])
        # Missing P1
        
        with self.assertRaises(ValueError):
            ProjectiveMeasurement([P0])

if __name__ == '__main__':
    unittest.main()
