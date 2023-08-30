from aqua.components.optimizers import COBYLA

from solver.solver import Solver


class COBYLASolver(Solver):
    def __init__(self, maxeval):
        super().__init__()
        self.optimiser = COBYLA(disp=True, maxiter=maxeval, tol=0)
