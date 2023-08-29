from unittest import TestCase

from qiskit.circuit import Instruction

from quantum_ansatz.ryrz_ansatz import RyRzAnsatz


class TestRyRzAnsatz(TestCase):
    def setUp(self):
        self.ansatz = RyRzAnsatz(4, 1, False)

    def test_create_circuit(self):
        qc = self.ansatz.circuit
        self.assertIsInstance(qc, Instruction)
        self.assertEqual(qc.num_qubits, 4)
        self.assertEqual(len(self.ansatz.params), len(qc.params))

