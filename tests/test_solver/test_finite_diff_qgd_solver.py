from unittest import TestCase

import numpy as np

from cost_function.n_coupled_qho_func import NCoupledQHOFunc
from solver.finite_diff_qgd_solver import FiniteDifferenceQGDSolver


class TestFiniteDiffQGDSolver(TestCase):
    def setUp(self) -> None:
        self.default_test_args = {"num_oscillators": 2,
                                  "encoding": "bin",
                                  "ansatz": "vatan-red",
                                  "num_qubits": 2,
                                  "depth": 1,
                                  "gammas": np.array([[0, 1.], [1., 0.]]),
                                  "optimiser": "aqgd-fin-diff",
                                  "shots": 10,
                                  "backend": "qasm_simulator",
                                  "noise_model": None}

        self.cost_func = NCoupledQHOFunc(**self.default_test_args)

    def test_call(self):
        solver = FiniteDifferenceQGDSolver(2, 0.25, 0.25)
        x, minf = solver(self.cost_func, np.ones(len(self.cost_func.all_params)), save=False)
        self.assertIsInstance(x, np.ndarray)
        self.assertEqual(x.shape, (len(self.cost_func.all_params),))
        self.assertIsInstance(minf, float)
