import numpy as np
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit import Parameter

from quantum_ansatz.quantum_ansatz import QuantumAnsatz


class SimAnsatz(QuantumAnsatz):
    """
    Variational ansatz based on circuit 14 from Adv. Quantum Technol. 2 (2019) 1900070 (arXiv:1905.10876).
    """

    def __init__(self, num_qubits: int, depth: int):
        super().__init__(num_qubits, depth)

        self.params = []
        self.num_qubits = num_qubits

        self.q = QuantumRegister(self.num_qubits)
        self.param_counter = 0

    def _create_circuit(self):
        qc = QuantumCircuit(self.q, name='QuantumAnsatz')
        for layer_no in range(self.depth):
            qc = self._add_rotations(qc)
            if layer_no % 2 == 1:
                qc = self._add_entangling_gates(qc, 3, 0)
            else:
                qc = self._add_entangling_gates(qc, 1, 1, reverse=True)
            qc.barrier()
        return qc.to_instruction()

    def _add_rotations(self, qc):
        for i in range(0, qc.num_qubits):
            param = Parameter("p" + self.param_counter.__str__())
            self.params += [param, ]
            qc.ry(param, self.q[i])
            self.param_counter += 1
        return qc

    def _add_entangling_gates(self, qc, ctrl_range, offset, reverse=False):
        for i in reversed(range(qc.num_qubits)) if reverse else range(qc.num_qubits):
            param = Parameter("p" + self.param_counter.__str__())
            self.params += [param, ]
            ctrl = (i + offset) % qc.num_qubits
            tgt = (i + offset + ctrl_range) % qc.num_qubits
            qc.crx(param, self.q[ctrl], self.q[tgt])
            self.param_counter += 1
        return qc

    def get_initial(self, initial_state='random'):
        self.circuit  # create circuit to populate self.params

        if initial_state == 'zero':
            initial_values = np.zeros(len(self.params))

        elif initial_state == 'random':
            initial_values = np.random.uniform(-np.pi, np.pi, len(self.params))

        else:
            raise ValueError('initial_state must be one of equal, zero or random.')

        return initial_values

    def get_phase(self, x):
        return 0
