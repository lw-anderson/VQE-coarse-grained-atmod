from unittest import TestCase

from solver.midaco_solver import MidacoSolver


class TestMidacoSolver(TestCase):
    def test_not_implemented(self):
        self.assertRaises(NotImplementedError, lambda: MidacoSolver())
