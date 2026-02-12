import pytest
import copy
import pickle
from src.core.quantum_object import QuantumObject
from src.core.exceptions import QuantumNoCloneError, QuantumCollapsedError, LineageDoubleSpendError
from src.core.lineage import LineageRegistry

@pytest.fixture(autouse=True)
def reset_lineage():
    LineageRegistry.reset_registry()

def test_no_cloning_shallow():
    """Verify that shallow copy attempts fail."""
    q_obj = QuantumObject("Secret Data")
    with pytest.raises(QuantumNoCloneError):
        copy.copy(q_obj)

def test_no_cloning_deep():
    """Verify that deep copy attempts fail."""
    q_obj = QuantumObject("Secret Data")
    with pytest.raises(QuantumNoCloneError):
        copy.deepcopy(q_obj)

def test_no_cloning_pickle():
    """Verify that serialization (cloning via persistence) fails."""
    q_obj = QuantumObject("Secret Data")
    with pytest.raises(QuantumNoCloneError):
        pickle.dumps(q_obj)

def test_measurement_collapse():
    """Verify that measurement consumes the state."""
    q_obj = QuantumObject("Data")
    
    # First measurement succeeds
    data = q_obj.measure()
    assert data == "Data"
    
    # Second measurement fails (Collapse)
    with pytest.raises(QuantumCollapsedError):
        q_obj.measure()

def test_double_spend_lineage_protection():
    """
    Simulate a 'hacker' trying to bypass object-level state 
    by extracting the ID and replaying it elsewhere.
    """
    q_obj = QuantumObject("Data")
    q_obj_id = q_obj.id
    
    # Legitimate consumption
    q_obj.measure()
    
    # Global Registry Check
    assert LineageRegistry.is_spent(q_obj_id)
    
    # Try to manually force-consume the ID again
    with pytest.raises(LineageDoubleSpendError):
        LineageRegistry.register_consumption(q_obj_id)

def test_information_loss_irreversibility():
    """
    Verify that once measured, the original object no longer holds data.
    """
    q_obj = QuantumObject("Ephemeral")
    q_obj.measure()
    
    # Internal state should be wiped
    assert q_obj._data is None
