import warnings

import numpy as np
from numpy.typing import NDArray
from qiskit import Aer, assemble
from scipy.sparse import csr_matrix

from cost_function.cost_func import CostFunc


class NCoupledQHOStatevectorFunc(CostFunc):

    def name(self):
        pass

    def __init__(self, num_oscillators: int,
                 encoding: str,
                 ansatz: str,
                 num_qubits: int,
                 depth: int,
                 gammas: NDArray,
                 optimiser: str,
                 save_temp_evals: bool = False,
                 anharmonic: dict = {"cubic": 0.0, "quartic": 0.0, "field": 0.0}):
        """
        Cost function object for system of N coupled oscillators, simulated using qiskit statevector simulator.
        Includes functionality for creating ansatz circuits, evaluating cost function expectation value.

        num_oscillators: number of oscillators N.
        encoding: Fock to qubit encoding, either 'bin' or 'gray'.
        ansatz: Ansatz type, see QuantumAnsatzFactory for value values.
        num_qubits: number of qubits. must be integer multiple of num_oscillators.
        depth: ansatz depth.
        gammas: NxN array where of diagonal terms correspond to coupling between oscillators.
        optimiser: type of optimiser to use, should be one of 'midaco', 'adam', 'aqgd', 'aqgd-fin-diff', 'cobyla'.
        Determines form of expectation output to match those required for (gradient and non-gradient based) optimisers.
        save_temp_evals: if True will save value of every expectation value to run directory.
        anharmonic: dict containing keys 'cubic', 'quartic', 'field' to add in anharmonic terms.
        """

        super().__init__(num_oscillators, encoding, ansatz, num_qubits, depth, gammas, optimiser, shots=None,
                         backend='statevector_simulator', noise_model=None, allow_dynamic_shots=False, lanczos=False,
                         meas_cal=False, save_temp_evals=save_temp_evals, anharmonic=anharmonic)

        warnings.warn("NCoupledQHOStatevectorFunc is now deprecated. Used NCoupledQHOFund instead"
                      "with backeend='statevector_simulator argument. NCoupledQHOStatevectorFunc may not "
                      "work as expected (or at all!).", DeprecationWarning)

    def evaluate_cost_function(self, all_params, all_param_indices):
        """
        Calculate cost function for Hamiltonian for coupled oscillators using expectations of Fock states |n> endcoding
        in binary states of qubits.
        """
        outputs = []
        for x, index in zip(all_params, all_param_indices):
            para_dic = {k: l for k, l in zip(self.all_params, x)}

            state_simulator = Aer.get_backend('statevector_simulator')
            qobj = assemble(self.transpiled_l.bind_parameters(para_dic), backend=state_simulator)
            job = state_simulator.run(qobj)
            state = np.array(job.result().get_statevector()).T
            if type(self.hamiltonian) is np.ndarray:
                expectation = np.dot(state.conj().T, np.dot(self.hamiltonian, state))
            elif type(self.hamiltonian) is csr_matrix:
                expectation = self.hamiltonian.dot(state).conj().transpose().dot(state)
            else:
                raise TypeError(f'expectation should be type {type(np.ndarray)}')
            expectation = np.real(expectation)
            outputs.append((expectation, 0))

        assert len(outputs) == len(all_params)

        return outputs
