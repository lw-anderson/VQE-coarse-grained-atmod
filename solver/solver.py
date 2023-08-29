from abc import ABC

import numpy as np


class Solver(ABC):
    def __init__(self):
        self.optimiser = None

    def __call__(self, problem_func, loc_vals, save=True):
        x, minf, evals = self.optimiser.optimize(num_vars=len(loc_vals), objective_function=problem_func,
                                                 initial_point=loc_vals)

        # saving to total list
        if save:
            np.savetxt("costfunc_values_final.txt", problem_func.costfunc_values)
            np.savetxt("parameters_values_final.txt", problem_func.parameter_values)

        return x, minf
