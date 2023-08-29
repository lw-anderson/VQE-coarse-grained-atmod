from abc import ABC
from typing import Union

from cost_function.n_coupled_qho_func import NCoupledQHOFunc
from cost_function.n_coupled_qho_statevector_func import NCoupledQHOStatevectorFunc
from utils import gammas_list_to_matrix


class CostFuncFactory(ABC):
    def __init__(self, args, save_temp_evals=True):
        self.backend = args.backend
        self.ansatz = args.ansatz
        self.num_qubits = args.num_qubits
        self.num_oscillators = args.num_oscillators
        self.anharmonic = {"cubic": args.cubic, "quartic": args.quartic,
                           "field": args.ext_field}
        self.encoding = args.encoding
        self.depth = args.depth
        self.shots = args.shots
        self.noise = args.noise
        self.backend = args.backend
        self.solver = args.solver
        self.dynamic_shots = args.dynamic_shots
        self.meas_cal = args.meas_cal
        self.save_temp_evals = save_temp_evals
        if type(args.gammas) == str:
            args.gammas = eval(args.gammas)
        assert len(args.gammas) == int(self.num_oscillators * (self.num_oscillators - 1) / 2), \
            ValueError('gammas_list is incorrect size.')

        self.gammas_matrix = gammas_list_to_matrix(args.gammas)

    def get(self) -> Union[NCoupledQHOStatevectorFunc, NCoupledQHOFunc]:
        return NCoupledQHOFunc(self.num_oscillators, self.encoding, self.ansatz, self.num_qubits, self.depth,
                               self.gammas_matrix, self.solver, self.shots, self.backend, self.noise, self.meas_cal,
                               self.save_temp_evals, self.anharmonic)
