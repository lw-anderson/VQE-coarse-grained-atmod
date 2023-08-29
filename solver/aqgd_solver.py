from qiskit.aqua.components.optimizers import AQGD

from solver.solver import Solver


class AQGDSolver(Solver):
    def __init__(self, maxeval, eta=0.25, momentum=0.25):
        super().__init__()
        self.optimiser = AQGD(disp=True, maxiter=maxeval, eta=eta, momentum=momentum, tol=0, param_tol=0)
