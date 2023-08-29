from unittest import TestCase

from qiskit.circuit import Instruction

from quantum_ansatz.vatan_ansatz import VatanAnsatz
from quantum_ansatz.vatan_reduced_ansatz import VatanReducedAnsatz


class TestVatanReducedAnsatz(TestCase):
    def setUp(self):
        self.ansatz = VatanReducedAnsatz(4, 1)

    def test_create_circuit(self):
        qc = self.ansatz.circuit
        self.assertIsInstance(qc, Instruction)
        self.assertEqual(qc.num_qubits, 4)
        self.assertEqual(len(self.ansatz.params), len(qc.params))

        vatan_orig_ansatz = VatanAnsatz(4, 1)
        vatan_orig_qc = vatan_orig_ansatz.circuit
        self.assertNotEquals(len(vatan_orig_ansatz.params), len(self.ansatz.params))
