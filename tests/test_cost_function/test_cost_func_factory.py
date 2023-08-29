from copy import copy
from unittest import TestCase

from cost_function.cost_func_factory import CostFuncFactory
from cost_function.n_coupled_qho_func import NCoupledQHOFunc
from optimisation_parser import OptimisationParser


class TestCostFuncFactory(TestCase):
    def setUp(self):
        parser = OptimisationParser()
        self.default_args = parser.parse_args()
        # self.default_args = OptimisationParser()

    def test_get_qasm_cost_func(self):
        fact = CostFuncFactory(self.default_args, False)
        cost_func = fact.get()
        self.assertIsInstance(cost_func, NCoupledQHOFunc)

    def test_get_statevector_cost_func(self):
        args = copy(self.default_args)
        args.backend = "statevector_simulator"
        fact = CostFuncFactory(args, False)
        cost_func = fact.get()
        self.assertIsInstance(cost_func, NCoupledQHOFunc)
