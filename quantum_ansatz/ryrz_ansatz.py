import numpy as np
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit import Parameter

from quantum_ansatz.quantum_ansatz import QuantumAnsatz


class RyRzAnsatz(QuantumAnsatz):
    """
    Variational ansatz described in Quantum 5, 492 (2021).
    """

    def __init__(self, num_qubits: int, depth: int, cyclic: bool = False):
        super().__init__(num_qubits, depth)
        self.num_qubits = num_qubits
        self.cyclic = cyclic
        self.depth = depth
        num_params = 2 * (1 + depth) * num_qubits

        self.params = [Parameter("p" + n.__str__()) for n in range(num_params)]

    def _create_circuit(self):
        params = iter(self.params)

        q = QuantumRegister(self.num_qubits)
        circuit = QuantumCircuit(q, name='QuantumAnsatz')

        for i in range(self.num_qubits):
            circuit.ry(next(params), i)
            circuit.rz(next(params), i)

        for layer in range(self.depth):
            for ctrl in range(self.num_qubits) if self.cyclic else range(self.num_qubits - 1):
                tgt = (ctrl + 1) % self.num_qubits
                circuit.cz(ctrl, tgt)
            for i in range(self.num_qubits):
                circuit.ry(next(params), i)
                circuit.rz(next(params), i)

        return circuit.to_instruction()

    def get_initial(self, initial_state='random'):

        self.circuit

        if initial_state == 'zero':
            initial_values = np.zeros(len(self.params))

        elif initial_state == 'random':
            initial_values = np.random.uniform(-np.pi, np.pi, len(self.params))

        else:
            raise ValueError('initial_state must be one of equal, zero or random.')

        return initial_values

    def get_phase(self, x):
        return 0
