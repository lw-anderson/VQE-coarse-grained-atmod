from quantum_ansatz.quantum_ansatz import QuantumAnsatz
from quantum_ansatz.ryrz_ansatz import RyRzAnsatz
from quantum_ansatz.sim_ansatz import SimAnsatz
from quantum_ansatz.vatan_ansatz import VatanAnsatz
from quantum_ansatz.vatan_reduced_ansatz import VatanReducedAnsatz


class QuantumAnsatzFactory:
    def __init__(self, ansatz: str, num_qubits: int, depth: int, num_oscillators: int):
        self.ansatz = ansatz
        self.num_qubits = num_qubits
        self.num_oscillators = num_oscillators
        self.depth = depth

        if type(ansatz) is not str:
            raise TypeError("ansatz should be string")
        if type(num_qubits) is not int or num_qubits < 0:
            raise TypeError("num_qubits must be positive int")
        if type(num_oscillators) is not int or num_oscillators < 0:
            if num_oscillators is not None:
                raise TypeError("num_oscillators must be positive int or None")
        if type(depth) is not int or depth < 0:
            raise TypeError("depth must be positive int")

    def get(self) -> QuantumAnsatz:
        if self.ansatz == 'vatan':
            return VatanAnsatz(self.num_qubits, self.depth)
        elif self.ansatz == 'vatan-red':
            return VatanReducedAnsatz(self.num_qubits, self.depth)
        elif self.ansatz == 'sim':
            return SimAnsatz(self.num_qubits, self.depth)
        elif self.ansatz == 'ryrz':
            return RyRzAnsatz(self.num_qubits, self.depth, cyclic=False)
        elif self.ansatz == 'ryrz-cyclic':
            return RyRzAnsatz(self.num_qubits, self.depth, cyclic=True)
        else:
            raise ValueError('Invalid ansatz type')
