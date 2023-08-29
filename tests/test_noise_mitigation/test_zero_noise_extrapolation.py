from unittest import TestCase

from qiskit import QuantumCircuit

from noise_mitigation.zero_noise_extrapolation import repeat_cnots, fold_circuit


class TestZeroNoiseExtrapolation(TestCase):
    def setUp(self) -> None:
        self.qc = QuantumCircuit(2)
        self.qc.h(0)
        self.qc.cnot(0, 1)
        self.qc.h(0)
        self.qc.x(1)

    def test_repeat_cnots(self):
        qc_cnots = repeat_cnots(self.qc, 3)
        self.assertEqual(len(qc_cnots), 9)
        self.assertRaises(ValueError, lambda: repeat_cnots(self.qc, 2))

    def test_fold_circuit(self):
        qc_folds = fold_circuit(self.qc, 3)
        self.assertEqual(len(qc_folds), 14)
        self.assertRaises(ValueError, lambda: fold_circuit(self.qc, 2))
