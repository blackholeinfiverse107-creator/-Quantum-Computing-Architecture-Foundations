import copy
import uuid
import sys
from .exceptions import QuantumNoCloneError, QuantumCollapsedError
from .lineage import LineageRegistry

class QuantumObject:
    """
    Core primitive that simulates quantum behavior:
    1. No-Cloning: copy.copy() raises error.
    2. Measurement Disturbance: measure() destroys the internal state.
    3. No-Deleting: (Attempts to track information loss if deleted unmeasured - simplified here).
    """

    def __init__(self, data):
        self._data = data
        self._id = str(uuid.uuid4())
        self._measured = False
        self._lineage_consumed = False

    @property
    def id(self):
        return self._id

    def measure(self):
        """
        Simulate a measurement operation.
        1. Checks if already collapsed/measured.
        2. Consumes the Lineage ID (prevents double measurement elsewhere).
        3. Returns the data.
        4. Destroys the internal reference.
        """
        if self._measured:
            raise QuantumCollapsedError("State has already collapsed due to previous measurement.")
        
        # Lineage Check - Global Registry
        # This catches "clones" that might have slipped past object-level checks (e.g. by memory manipulation or serialization hacks)
        LineageRegistry.register_consumption(self._id)
        
        # Collapse
        self._measured = True
        result = self._data
        
        # Information Destruction
        self._data = None 
        
        return result

    def __copy__(self):
        """Prevent shallow copy."""
        raise QuantumNoCloneError("Quantum state cannot be cloned.")

    def __deepcopy__(self, memo):
        """Prevent deep copy."""
        raise QuantumNoCloneError("Quantum state cannot be cloned.")

    def __getstate__(self):
        """
        Prevent serialization (pickling) by default unless strict transfer protocols are used.
        For this prototype, we simply ban it as 'cloning via persistence'.
        """
        raise QuantumNoCloneError("Serialization constitutes cloning/measurement without authorization.")

    def __del__(self):
        """
        Destructor hook.
        In a real quantum system, information is conserved.
        Deleting an unmeasured quantum state implies information loss (entropy increase).
        We can log this as a 'Loss' event, though we can't stop Python from GC'ing.
        """
        # This is a best-effort check for the prototype.
        if not self._measured and not self._lineage_consumed:
            # In a strict system, this might trigger a system-wide alert.
            # print(f"WARNING: Quantum Information Loss detected for ID {self._id}")
            pass
