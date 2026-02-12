class QuantumNoCloneError(Exception):
    """Raised when an attempt is made to copy a QuantumObject."""
    pass

class QuantumCollapsedError(Exception):
    """Raised when an attempt is made to access a collapsed (measured) QuantumObject."""
    pass

class LineageDoubleSpendError(Exception):
    """Raised when a unique lineage ID is reused."""
    pass
