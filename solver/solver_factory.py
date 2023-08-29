from solver.adam_solver import ADAMSolver
from solver.aqgd_solver import AQGDSolver
from solver.cobyla_solver import COBYLASolver
from solver.finite_diff_qgd_solver import FiniteDifferenceQGDSolver


class SolverFactory:
    def __init__(self, args):
        self.args = args

    def get(self):
        if self.args.solver == 'midaco':
            raise NotImplementedError("Midaco solver not supported in open access version (Needs subscription key).")
        elif self.args.solver == 'aqgd':
            solver = AQGDSolver(maxeval=self.args.maxeval, eta=self.args.eta, momentum=self.args.momentum)
        elif self.args.solver == 'aqgd-fin-diff':
            solver = FiniteDifferenceQGDSolver(maxeval=self.args.maxeval, eta=self.args.eta,
                                               momentum=self.args.momentum)
        elif self.args.solver == 'adam':
            solver = ADAMSolver(maxeval=self.args.maxeval, lr=self.args.eta)
        elif self.args.solver == 'cobyla':
            solver = COBYLASolver(maxeval=self.args.maxeval)
        else:
            raise ValueError('Invalid solver.')

        return solver
