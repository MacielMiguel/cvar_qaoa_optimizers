"""
Abstract Hamiltonian interface.

Any problem Hamiltonian passed to quantum_core.experiment must subclass this.
Implementing to_pauli_op() is required to run QAOA/VQE experiments.
"""
from abc import ABC, abstractmethod
from typing import Dict


class Hamiltonian(ABC):
    """Base class for Hamiltonian specifications."""

    num_qubits: int

    @abstractmethod
    def evaluate(self, bitstring: str) -> float:
        """Evaluate energy for a given bitstring."""
        pass

    def to_pauli_op(self):
        """
        Return a Qiskit SparsePauliOp representation of this Hamiltonian.

        The result is cached on the instance after the first call — repeated
        calls (e.g. across multiple run_experiment() calls in a suite) are free.

        Required to run QAOA/VQE experiments. Raises NotImplementedError
        for Hamiltonians that have not implemented this conversion.
        """
        if not hasattr(self, "_pauli_op_cache"):
            self._pauli_op_cache = self._build_pauli_op()
        return self._pauli_op_cache

    def _build_pauli_op(self):
        """Override this method in subclasses instead of to_pauli_op()."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement _build_pauli_op(). "
            "Add this method to use the Hamiltonian with quantum_core.experiment."
        )

    @abstractmethod
    def to_dict(self) -> Dict:
        """Serialise to a plain dict (used by results_io)."""
        pass
