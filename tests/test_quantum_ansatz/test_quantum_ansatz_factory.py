from unittest import TestCase

from quantum_ansatz.quantum_ansatz_factory import QuantumAnsatzFactory
from quantum_ansatz.ryrz_ansatz import RyRzAnsatz
from quantum_ansatz.sim_ansatz import SimAnsatz
from quantum_ansatz.vatan_ansatz import VatanAnsatz
from quantum_ansatz.vatan_reduced_ansatz import VatanReducedAnsatz


class TestQuantumAnsatzFactory(TestCase):
    def test_get_success(self):
        args = {"num_qubits": 4, "depth": 2, "num_oscillators": 2}

        vatan_ansatz = QuantumAnsatzFactory("vatan", **args).get()
        self.assertIsInstance(vatan_ansatz, VatanAnsatz)

        vatan_reduced_ansatz = QuantumAnsatzFactory("vatan-red", **args).get()
        self.assertIsInstance(vatan_reduced_ansatz, VatanReducedAnsatz)

        vatan_reduced_ansatz = QuantumAnsatzFactory("sim", **args).get()
        self.assertIsInstance(vatan_reduced_ansatz, SimAnsatz)

        vatan_reduced_ansatz = QuantumAnsatzFactory("ryrz", **args).get()
        self.assertIsInstance(vatan_reduced_ansatz, RyRzAnsatz)

        vatan_reduced_ansatz = QuantumAnsatzFactory("ryrz-cyclic", **args).get()
        self.assertIsInstance(vatan_reduced_ansatz, RyRzAnsatz)

    def test_invalid_args(self):
        self.assertRaises(TypeError, lambda: QuantumAnsatzFactory(1, 4, 1, 2))
        self.assertRaises(TypeError, lambda: QuantumAnsatzFactory("vatan", "wrong qubits", 1, 2))
        self.assertRaises(TypeError, lambda: QuantumAnsatzFactory("vatan", 4, "wring depth", 2))
        self.assertRaises(TypeError, lambda: QuantumAnsatzFactory("vatan", 4, 1, "wrong oscillators"))
        self.assertRaises(ValueError, QuantumAnsatzFactory("invalid ansatz", 4, 1, 2).get)
