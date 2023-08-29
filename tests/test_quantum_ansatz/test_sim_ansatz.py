from unittest import TestCase

from qiskit.circuit import Instruction

from quantum_ansatz.sim_ansatz import SimAnsatz


class TestRyRzAnsatz(TestCase):
    def setUp(self):
        self.ansatz = SimAnsatz(4, 1)

    def test_create_circuit(self):
        qc = self.ansatz.circuit
        self.assertIsInstance(qc, Instruction)
        self.assertEqual(qc.num_qubits, 4)
        self.assertEqual(len(self.ansatz.params), len(qc.params))
