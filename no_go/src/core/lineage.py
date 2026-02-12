from typing import Set
from .exceptions import LineageDoubleSpendError

class LineageRegistry:
    """
    Global registry to track the lineage of QuantumObjects.
    Enforces:
    1. No-Double-Spend: An ID can only be consumed once.
    2. No-Rollback: History is append-only (simulated by non-removable entries).
    """
    _spent_ids: Set[str] = set()

    @classmethod
    def register_consumption(cls, unique_id: str) -> None:
        """
        Mark a unique ID as consumed/spent.
        Raises LineageDoubleSpendError if already spent.
        """
        if unique_id in cls._spent_ids:
            raise LineageDoubleSpendError(f"Lineage ID {unique_id} has already been spent. Double-spend attempt detected.")
        cls._spent_ids.add(unique_id)

    @classmethod
    def is_spent(cls, unique_id: str) -> bool:
        """Check if an ID is already spent."""
        return unique_id in cls._spent_ids

    @classmethod
    def reset_registry(cls):
        """
        Debug/Test only: Reset the registry. 
        In a real physical system, this would be impossible (creation of a new universe).
        """
        cls._spent_ids.clear()
