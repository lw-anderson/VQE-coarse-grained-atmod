from unittest import TestCase

import numpy as np

from cost_function.n_coupled_qho_statevector_func import NCoupledQHOStatevectorFunc


class TestNCoupledQHOStatevectorFunc(TestCase):
    def test_init(self) -> None:
        self.default_test_args = {"num_oscillators": 2,
                                  "encoding": "bin",
                                  "ansatz": "vatan-red",
                                  "num_qubits": 4,
                                  "depth": 1,
                                  "gammas": np.array([[0, 1.], [1., 0.]]),
                                  "optimiser": "adam",
                                  }

        self.assertWarns(DeprecationWarning, lambda: NCoupledQHOStatevectorFunc(**self.default_test_args))
