from abc import ABC, abstractmethod

import numpy as np
from qiskit import QuantumCircuit


class QuantumAnsatz(ABC):

    def __init__(self, num_qubits: int, depth: int):
        """
        Base class for quantum ansatz.
        num_qubits: number of qubits for ansatz
        depth: number of layers of ansatz, definition of layer will vary by ansatz type.
        """

        # Check that num_qubits is an integer
        try:
            num_qubits = int(num_qubits)
        except (NameError, ValueError):
            print("Number of qubits must be positive integer.")

        if num_qubits == 0:
            raise ValueError("Number of qubits must be positive integer.")

        self.num_qubits = num_qubits

        try:
            depth = int(depth)
        except (NameError, ValueError):
            print("Depth must be positive integer.")

        if depth == 0:
            raise ValueError("Depth must be positive integer.")

        self.depth = depth

        self._circuit = None

    @abstractmethod
    def _create_circuit(self) -> QuantumCircuit:
        pass

    @property
    def circuit(self) -> QuantumCircuit:
        if self._circuit is None:
            self._circuit = self._create_circuit()
        return self._circuit

    @abstractmethod
    def get_initial(self, initial_state='equal') -> np.ndarray:
        pass
