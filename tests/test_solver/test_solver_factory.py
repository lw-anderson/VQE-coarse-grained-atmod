from unittest import TestCase

from optimisation_parser import OptimisationParser
from solver.adam_solver import ADAMSolver
from solver.aqgd_solver import AQGDSolver
from solver.cobyla_solver import COBYLASolver
from solver.finite_diff_qgd_solver import FiniteDifferenceQGDSolver
from solver.solver_factory import SolverFactory


class TestSolverFactory(TestCase):
    def test_get(self):
        parser = OptimisationParser()
        args = parser.parse_args()

        args.solver = "midaco"
        self.assertRaises(NotImplementedError, SolverFactory(args).get)

        args.solver = "aqgd"
        self.assertIsInstance(SolverFactory(args).get(), AQGDSolver)

        args.solver = "aqgd-fin-diff"
        self.assertIsInstance(SolverFactory(args).get(), FiniteDifferenceQGDSolver)

        args.solver = "adam"
        self.assertIsInstance(SolverFactory(args).get(), ADAMSolver)

        args.solver = "cobyla"
        self.assertIsInstance(SolverFactory(args).get(), COBYLASolver)
