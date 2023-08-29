import functools
import logging
import os
import warnings
from abc import ABC
from copy import copy
from datetime import datetime
from itertools import chain, repeat
from typing import List, Union, Tuple, Dict, Any

import numpy as np
from numpy.typing import NDArray
from qiskit import Aer, IBMQ, transpile, QuantumRegister, QuantumCircuit
from qiskit.aqua import AquaError
from qiskit.aqua.operators import WeightedPauliOperator, TPBGroupedWeightedPauliOperator
from qiskit.aqua.operators.legacy import op_converter
from qiskit.compiler import assemble
from qiskit.ignis.mitigation import complete_meas_cal, CompleteMeasFitter
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.ibmq import IBMQJobManager, IBMQBackend
from qiskit.quantum_info import Pauli
from qiskit.utils import QuantumInstance
from qiskit.utils.backend_utils import is_statevector_backend

from measure_commuting_operators import one_factorisation
from measure_commuting_operators.sorted_insertion import sorted_insertion
from noise_mitigation.zero_noise_extrapolation import repeat_cnots, fold_circuit
from operators.calculate_pauli_coefficients import calculate_pauli_coefficient_non_interaction, \
    calculate_pauli_coefficient_coupled_pair, append_indices_and_coeffs_to_list, calculate_non_zero_terms
from operators.hamiltonian import get_extended_hamiltonian
from quantum_ansatz.quantum_ansatz_factory import QuantumAnsatzFactory
from utils import get_qubit_mapping_from_transpiled_circuit

os.environ['QISKIT_IN_PARALLEL'] = 'TRUE'


class NCoupledQHOFunc(ABC):
    def __init__(self, num_oscillators: int,
                 encoding: str,
                 ansatz: str,
                 num_qubits: int,
                 depth: int,
                 gammas: NDArray,
                 optimiser: str,
                 shots: int,
                 backend: str,
                 noise_model: str,
                 # allow_dynamic_shots: bool = False,
                 # lanczos: bool = False,
                 meas_cal: bool = False,
                 save_temp_evals: bool = False,
                 anharmonic: dict = {"cubic": 0.0, "quartic": 0.0, "field": 0.0}):
        """
        Cost function object for system of N coupled oscillators. In functionality for creating ansatz circuits,
        evaluating cost function expectation values (including H, H^2 and H^3) as well as performing Lanczos+ZNE error
        mitigation and noisy-uncoupled subtraction.

        num_oscillators: number of oscillators N.
        encoding: Fock to qubit encoding, either 'bin' or 'gray'.
        ansatz: Ansatz type, see QuantumAnsatzFactory for value values.
        num_qubits: number of qubits. must be integer multiple of num_oscillators.
        depth: ansatz depth.
        gammas: NxN array where of diagonal terms correspond to coupling between oscillators.
        optimiser: type of optimiser to use, should be one of 'midaco', 'adam', 'aqgd', 'aqgd-fin-diff', 'cobyla'.
        Determines form of expectation output to match those required for (gradient and non-gradient based) optimisers.
        shots: number of shots for each expectation evaluation
        backend: string determining type of backend use, should be one of 'statevector_simulator', 'qasm_simulator',
        'fake_montreal' or 'fake_lagos'.
        repeatedly probe IBMQ servers to find real backend matching name.
        noise_model: string determining (fake or real) noise profile/backend for simulation. Should be one of
        'fake_mumbai', 'fake_montreal' or 'load' (need backend object saved in run directory).
        meas_cal: if True will perform measurement calculation on instantiation and then use this to correct
        expectation values.
        save_temp_evals: if True will save value of every expectation value to run directory.
        anharmonic: dict containing keys 'cubic', 'quartic', 'field' to add in anharmonic terms.
        """

        self.save_temp_evals = save_temp_evals
        if save_temp_evals:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.temp_outputs_fname = 'temp-evals ' + now + '.npy'
            self.temp_params_fname = 'temp-params ' + now + '.npy'

        if num_oscillators < 1 or num_oscillators > 6:
            raise ValueError("num_oscillators must be an integer between 2 and 6 (inclusive).")

        if num_qubits % num_oscillators != 0:
            raise ValueError("num_qubits must be a multiple of num_oscillators.")

        # Backend configuration
        if noise_model:
            print('noise model = ', noise_model)
            if noise_model == 'load':
                self.noise_backend = np.load('noise_backend.npy', allow_pickle=True)[0]
            elif noise_model == 'fake_mumbai':
                from qiskit.test.mock import FakeMumbai
                self.noise_backend = FakeMumbai()
            elif noise_model == 'fake_montreal':
                from qiskit.test.mock import FakeMontreal
                self.noise_backend = FakeMontreal()
            else:
                raise NotImplementedError(
                    "Invalid or incompatible noise model. Note: Real use of IBMQ backends has been disabled (requires "
                    "now lapsed IBM-Oxford access agreement)")
                provider = None
                while provider is None:
                    try:
                        IBMQ.load_account()
                        provider = IBMQ.get_provider(hub='ibm-q-research-2', group='uni-oxford-2',
                                                     project='main')
                    except Exception as e:
                        logging.warning(e)
                        logging.warning('Unable to get IBMQ provider, retrying.')
                self.noise_backend = provider.get_backend(noise_model)

            self.noise_model = NoiseModel.from_backend(self.noise_backend)

        if backend in ['statevector_simulator', 'qasm_simulator']:
            self.backend = Aer.get_backend(backend, noise_model=self.noise_model if noise_model else None)

        elif backend == 'fake_montreal':
            from qiskit.test.mock import FakeMontreal
            self.backend = FakeMontreal()

        elif backend == 'fake_lagos':
            from qiskit.test.mock import FakeLagos
            self.backend = FakeLagos()

        else:
            raise NotImplementedError("Invalid or incompatible backend. Note: Real use of IBMQ backends has been "
                                      "disabled (requires now lapsed IBM-Oxford access agreement)")
            IBMQ.load_account()
            provider = IBMQ.get_provider(hub='ibm-q-research-2', group='uni-oxford-2',
                                         project='main')
            self.backend = provider.get_backend(backend)

        print('Backend = ', self.backend, 'Noise model =', noise_model)

        if noise_model:
            self.quantum_instance = QuantumInstance(backend=self.backend, noise_model=self.noise_model,
                                                    coupling_map=self.backend.configuration().coupling_map,
                                                    shots=8192)
        else:
            self.quantum_instance = QuantumInstance(backend=self.backend,
                                                    coupling_map=self.backend.configuration().coupling_map,
                                                    shots=8192)

        # Cost function and run options

        self.N = num_oscillators
        self.n = num_qubits
        self.d = depth
        self.shots = shots
        self.gammas = gammas
        self.encoding = encoding

        self.hamiltonian = get_extended_hamiltonian(num_oscillators, num_qubits, gammas=gammas,
                                                    cubic_prefactor=anharmonic["cubic"],
                                                    quartic_prefactor=anharmonic["quartic"],
                                                    external_field=anharmonic["field"],
                                                    encoding=encoding)

        # self.lanczos = lanczos

        self.optimiser = optimiser

        self.grid = np.linspace(0, 1, 2 ** self.n, endpoint=False)
        self.costfunc_values = []
        self.parameter_values = []
        self.analytic_minimum = None

        non_int_groups_all_oscillators, non_int_groups_separate_oscillators \
            = self.create_groups_for_non_interacting_terms()
        coupling_groups_all_pairs, coupling_groups_separate_pairs \
            = self.create_groups_for_coupling_terms()

        self.ansatz = QuantumAnsatzFactory(ansatz, self.n, self.d, self.N).get()

        self.all_params = self.ansatz.params

        self.qr = QuantumRegister(self.n)
        self.qc_l = QuantumCircuit(self.qr)
        self.qc_l.append(self.ansatz.circuit, self.qc_l.qubits)
        self.qc_l = transpile(self.qc_l, optimization_level=3, basis_gates=['id', 'rz', 'sx', 'x', 'cx'])
        self.transpiled_l = transpile(self.qc_l, optimization_level=3, basis_gates=['id', 'rz', 'sx', 'x', 'cx'],
                                      backend=self.noise_backend if hasattr(self, 'noise_backend') else self.backend,
                                      layout_method='noise_adaptive')

        self.qubit_layout = get_qubit_mapping_from_transpiled_circuit(self.transpiled_l)
        print('Using physical qubits', self.qubit_layout)

        if meas_cal:
            self.meas_calibration_fitter = self.do_measurement_calibration()

        pauli_options = ['I', 'X', 'Y', 'Z']
        paulis = []

        if not (anharmonic["cubic"] or anharmonic["quartic"] or anharmonic["field"]):
            for group in non_int_groups_all_oscillators + coupling_groups_all_pairs:
                for (indices, coeff) in group:
                    pauli_string = ''
                    for ind in tuple(reversed(indices)):
                        pauli_string += pauli_options[ind]
                    paulis.append([coeff, Pauli(pauli_string)])

        else:
            all_indices, all_coefficients = calculate_non_zero_terms(self.hamiltonian, self.n)
            for (indices, coeff) in zip(all_indices, all_coefficients):
                pauli_string = ''
                for ind in tuple(reversed(indices)):
                    pauli_string += pauli_options[ind]
                paulis.append([coeff, Pauli(pauli_string)])

        h_operator = WeightedPauliOperator(paulis)

        if backend == "statevector_simulator":
            self.h_operator = self.hamiltonian
            self.h2_operator = self.hamiltonian @ self.hamiltonian
            self.h3_operator = self.h2_operator @ self.hamiltonian
        else:
            self.h_operator = op_converter.to_tpb_grouped_weighted_pauli_operator(h_operator,
                                                                                  TPBGroupedWeightedPauliOperator.sorted_grouping)
            self.h2_operator = op_converter.to_tpb_grouped_weighted_pauli_operator(h_operator * h_operator,
                                                                                   TPBGroupedWeightedPauliOperator.sorted_grouping)
            self.h3_operator = op_converter.to_tpb_grouped_weighted_pauli_operator(h_operator * h_operator * h_operator,
                                                                                   TPBGroupedWeightedPauliOperator.sorted_grouping)

    def create_groups_for_non_interacting_terms(self) -> Tuple[List[List], List[List]]:
        """
        Creates groups of coefficients and indices for the non interacting terms. All terms here act on different
        qubits and so will be returned as a single group.
        """

        one_osc_non_int_indices, one_osc_non_int_coeffs = calculate_pauli_coefficient_non_interaction(
            int(self.n / self.N), self.encoding)

        single_oscillator_identity_indices = (0,) * int(self.n / self.N)

        # Groups combining all oscillators together
        non_int_indices, non_int_coeffs = [], []
        for indices, coeff in zip(one_osc_non_int_indices, one_osc_non_int_coeffs):
            for i in range(self.N):
                indices1 = single_oscillator_identity_indices * i \
                           + indices \
                           + single_oscillator_identity_indices * (self.N - i - 1)
                non_int_indices, non_int_coeffs = append_indices_and_coeffs_to_list(indices1, coeff, non_int_indices,
                                                                                    non_int_coeffs)
        combined_oscillator_groups = [list(zip(non_int_indices, non_int_coeffs))]

        # Groups separated into hamiltonian terms for each oscillator
        individual_oscillator_groups = []
        for i in range(self.N):
            non_int_indices, non_int_coeffs = [], []
            for indices, coeff in zip(one_osc_non_int_indices, one_osc_non_int_coeffs):
                indices1 = single_oscillator_identity_indices * i \
                           + indices \
                           + single_oscillator_identity_indices * (self.N - i - 1)
                non_int_indices, non_int_coeffs = append_indices_and_coeffs_to_list(indices1, coeff, non_int_indices,
                                                                                    non_int_coeffs)
            individual_oscillator_groups.append([list(zip(non_int_indices, non_int_coeffs))])

        return combined_oscillator_groups, individual_oscillator_groups

    def create_groups_for_coupling_terms(self) -> Tuple[List[List], List[List]]:
        """
        Create groups of coefficients and indices for all coupling terms (for up to six oscillators). Using one
        factorisations of the coupling graphs terms acting on different oscillators are grouped together. There is no
        grouping of terms within each oscillator pair, this will be done by another method in the future.
        """

        if self.N == 1:
            return [], []

        pair_osc_coupled_indices, pair_osc_coupled_coeffs = calculate_pauli_coefficient_coupled_pair(
            int(2 * self.n / self.N), self.encoding)

        grouped_pairs = sorted_insertion(pair_osc_coupled_indices, pair_osc_coupled_coeffs, use=True)
        one_factorisation_groups = []

        gammas_matrix = self.gammas

        single_oscillator_identity_indices = (0,) * int(self.n / self.N)

        # Groups combining all interactions using 1-factorisations.
        for group in grouped_pairs:
            for covering in one_factorisation.k(self.N):

                one_factorisation_coeffs, one_factorisation_indices = [], []

                for (indices, coeff) in group:

                    for (i, j) in covering:
                        coeff2 = coeff * gammas_matrix[i, j]
                        indices2 = single_oscillator_identity_indices * i \
                                   + indices[:int(self.n / self.N)] \
                                   + single_oscillator_identity_indices * (j - i - 1) \
                                   + indices[int(self.n / self.N):] \
                                   + single_oscillator_identity_indices * (self.N - j - 1)

                        one_factorisation_indices, one_factorisation_coeffs \
                            = append_indices_and_coeffs_to_list(indices2, coeff2, one_factorisation_indices,
                                                                one_factorisation_coeffs)

                one_factorisation_groups.append(list(zip(one_factorisation_indices, one_factorisation_coeffs)))

        # Groups split into the separate two oscillator interactions.
        separate_interaction_groups = []
        for covering in one_factorisation.k(self.N):
            for (i, j) in covering:
                if gammas_matrix[i, j] != 0.0:
                    new_groups = []
                    for group in grouped_pairs:
                        individual_coupling_indices, individual_coupling_coeffs = [], []
                        for (indices, coeff) in group:
                            coeff2 = coeff * gammas_matrix[i, j]
                            indices2 = single_oscillator_identity_indices * i \
                                       + indices[:int(self.n / self.N)] \
                                       + single_oscillator_identity_indices * (j - i - 1) \
                                       + indices[int(self.n / self.N):] \
                                       + single_oscillator_identity_indices * (self.N - j - 1)
                            individual_coupling_indices, individual_coupling_coeffs \
                                = append_indices_and_coeffs_to_list(indices2, coeff2, individual_coupling_indices,
                                                                    individual_coupling_coeffs)

                        new_groups.append(list(zip(individual_coupling_indices, individual_coupling_coeffs)))

                    separate_interaction_groups.append(new_groups)

        return one_factorisation_groups, separate_interaction_groups

    def __call__(self, x: NDArray[float]):
        """
        Simulate circuit for set of parameter values (as will be done by optimiser) and return cost function in
        correct format for the optimiser.
        """

        if self.optimiser == 'midaco':
            # single evaluation (with not stdev output)
            cost_function = self.evaluate_cost_function([x], [self.h_operator])[0][0]
            return [cost_function], [0.0]

        elif self.optimiser in ['adam', 'aqgd', 'aqgd-fin-diff', 'cobyla']:
            # multiple evaluations at the same time in order to calculate cost function as well as gradients
            num_params = len(self.all_params)

            all_params = x.reshape(-1, num_params)
            output = self.evaluate_cost_function(all_params, self.h_operator)

            self.costfunc_values.append(output[0])
            self.parameter_values.append(x.reshape(-1, num_params)[0])
            if self.save_temp_evals:
                np.save(self.temp_outputs_fname, self.costfunc_values, allow_pickle=True)
                np.save(self.temp_params_fname, self.parameter_values, allow_pickle=True)

            if self.optimiser == 'adam':
                return output
            else:
                return [exp for (exp, stdev) in output]

        else:
            raise ValueError("Invalid optimiser type.")

    def construct_circuit_operator(self, circuit: QuantumCircuit,
                                   operator: Union[WeightedPauliOperator, np.ndarray],
                                   para_dic: Dict[str, float],
                                   backend: Union[IBMQBackend, None] = None,
                                   statevector_mode: Union[bool, None] = None,
                                   circuit_name_prefix: str = '',
                                   do_transpile: bool = True) -> List[QuantumCircuit]:
        """
        Given ansatz circuit, prepare list of circuits to calculate expectation value for operator, with bound
        parameters. Statevector mode to define if using statevector or shot based backemd, if backend supplied will be
        automatically chosen using backend type. Circuit name prefix used to label circuits to keep track of jobs when
        submitting multiple circuits as single job.
        operator must either be WeightedPauliOperator or numpy array (only in the case of statevector simulation).
        """

        if backend is not None:
            warnings.warn("backend option is deprecated and it will be removed after 0.6, "
                          "Use `statevector_mode` instead", DeprecationWarning)
            statevector_mode = is_statevector_backend(backend)
        else:
            if statevector_mode is None:
                raise AquaError("Either backend or statevector_mode need to be provided.")

        wave_function = circuit.bind_parameters(para_dic)

        if statevector_mode:
            circuit = copy(wave_function)
            circuit.name = circuit_name_prefix + 'psi'
            circuits = [circuit]
        else:
            circuits = operator.construct_evaluation_circuit(
                wave_function=wave_function,
                statevector_mode=statevector_mode, circuit_name_prefix=circuit_name_prefix, qr=wave_function.qregs[0])

            if do_transpile:
                circuits = transpile(circuits,
                                     backend=self.noise_backend if hasattr(self, 'noise_backend') else self.backend,
                                     initial_layout=self.qubit_layout)

        return circuits

    def evaluate_cost_function(self,
                               parameter_sets: NDArray[NDArray[float]],
                               operators: Union[WeightedPauliOperator, List[WeightedPauliOperator],
                               np.ndarray, List[np.ndarray]],
                               repeated_cnots: Union[List[int], int] = 1,
                               circuit_folds: Union[List[int], int] = 1,
                               return_zero_state_overlap: bool = False,
                               max_evals_at_once: int = 200) \
            -> Union[List[Tuple[float, float]], Tuple[List[Tuple[float, float]], List[float]]]:
        """
        Evaluate cost function (expectation value) for circuit.
        parameter_sets: list of numpy array each of length num_params. This corresponds to different values of
        parameters to evaluate expectation wrt. This allows for the batching of multiple params (for error mitigation
        and gradient calculation into one job)
        operators: operator to take expectation value, typically h1 and sometime h2,h3 for self. To evaluate multiple
        different operators, input as list of same length as parameter_sets. Should be qiskit operator or numpy array
        (only when using statevector simulator).
        repeated_cnots: number of times to repeat CNOTs for noise extrap. To evaluate multiple
        different values, input as list of same length as parameter_sets.
        circuit_folds: as above but with circuit folds for noise.
        return_zero_state_overlap: if true returns overlap with zero state as well as expectation values.
        max_evals_at_once: number of experiments batched into single job.
        """

        if type(operators) is list:
            assert len(operators) == len(parameter_sets), \
                'Number of operator objects should equal number of parameter sets (or a single operator object).'
        else:
            operators = [operators, ] * len(parameter_sets)

        if type(repeated_cnots) is list:
            assert len(repeated_cnots) == len(parameter_sets), \
                'Number of repeated_cnots values should equal number of parameter sets (or a single integer).'
        else:
            assert type(repeated_cnots) == int, \
                'Number of repeated_cnots values should equal number of parameter sets (or a single integer).'
            repeated_cnots = [repeated_cnots, ] * len(parameter_sets)

        if type(circuit_folds) is list:
            assert len(circuit_folds) == len(parameter_sets), \
                'Number of repeated_cnots values should equal number of parameter sets (or a single integer).'
        else:
            assert type(circuit_folds) == int, \
                'Number of circuit_folds values should equal number of parameter sets (or a single integer).'
            circuit_folds = [circuit_folds, ] * len(parameter_sets)

        noise_mitigated_circuits = {}
        for cnots in list(dict.fromkeys(repeated_cnots)):
            for folds in list(dict.fromkeys(circuit_folds)):
                noise_mitigated_circuits[str(cnots) + str(folds)] \
                    = repeat_cnots(fold_circuit(self.qc_l, folds), cnots)

        outputs = []
        zero_state_overlaps = []

        # if running on real device, circuits must be batched/split up to correct sizes to run as managed job set
        if isinstance(self.backend, IBMQBackend):
            circuits = []
            print(f'{datetime.now()}: constructing circuits')
            for idx, (x, operator, cnots, folds) \
                    in enumerate(zip(parameter_sets, operators, repeated_cnots, circuit_folds)):
                para_dic = {k: l for k, l in zip(self.all_params, x)}
                circuit = self.construct_circuit_operator(noise_mitigated_circuits[str(cnots) + str(folds)],
                                                          operator, para_dic,
                                                          statevector_mode=self.quantum_instance.is_statevector,
                                                          circuit_name_prefix=str(idx))
                circuits.append(circuit)

            circuits_to_be_run = functools.reduce(lambda x, y: x + y, circuits)

            job_manager = IBMQJobManager()
            print(f'{datetime.now()}: running circuits')
            shots = min([8192, self.backend.configuration().max_shots])
            managed_job_set = job_manager.run(circuits_to_be_run, max_experiments_per_job=max_evals_at_once,
                                              backend=self.backend, shots=shots)
            print('jobset id = ', managed_job_set.job_set_id())
            results = managed_job_set.results()

            results = results.combine_results()

            print(f'{datetime.now()}: run circuits')

            for idx, operator in enumerate(operators):
                mean, stdev = operator.evaluate_with_result(
                    result=results, statevector_mode=self.quantum_instance.is_statevector,
                    circuit_name_prefix=str(idx))
                outputs.append((np.real(mean), np.real(stdev)))

                if return_zero_state_overlap:
                    if Pauli("Z" * self.n) in [b[0] for b in operator.basis]:
                        zzzz_counts = results.get_counts(str(idx) + "Z" * self.n)
                        zero_overlap = zzzz_counts['0' * self.n] / sum(zzzz_counts.values())
                    else:
                        zero_overlap = None
                    zero_state_overlaps.append(zero_overlap)

            assert len(outputs) == len(parameter_sets), ValueError('Mismatch between input and output lengths.')

            if return_zero_state_overlap:
                return outputs, zero_state_overlaps

            return outputs

        # if using simulator (whether statevector or shot based), circuits are batched up into single job
        # according to max_evals_at_once.
        else:
            for offset_idx in range(0, len(parameter_sets), max_evals_at_once):
                subset_parameter_sets = parameter_sets[offset_idx:offset_idx + max_evals_at_once]
                subset_operators = operators[offset_idx:offset_idx + max_evals_at_once]
                subset_cnots = repeated_cnots[offset_idx:offset_idx + max_evals_at_once]
                subset_folds = circuit_folds[offset_idx:offset_idx + max_evals_at_once]
                circuits = []
                for idx, (x, operator, cnots, folds) \
                        in enumerate(zip(subset_parameter_sets, subset_operators, subset_cnots, subset_folds)):
                    assert len(self.all_params) == len(x), ValueError("dimension of x is not same as number of params.")

                    para_dic = {k: l for k, l in zip(self.all_params, x)}
                    circuit = self.construct_circuit_operator(noise_mitigated_circuits[str(cnots) + str(folds)],
                                                              operator, para_dic,
                                                              statevector_mode=self.quantum_instance.is_statevector,
                                                              circuit_name_prefix=str(idx + offset_idx),
                                                              do_transpile=not self.quantum_instance.is_statevector)
                    circuits.append(circuit)

                to_be_simulated_circuits = functools.reduce(lambda x, y: x + y, circuits)
                results = self.quantum_instance.execute(to_be_simulated_circuits, had_transpiled=True)

                if hasattr(self, 'meas_calibration_fitter'):
                    self.meas_calibration_fitter.filter.apply(results)

                for idx, operator in enumerate(subset_operators):

                    # for statevector simulator, each circuit is run a single time (as defined by
                    # self.construct_circuit_operator to give statevector.
                    if self.quantum_instance.is_statevector:
                        assert type(operator) is np.ndarray, TypeError(
                            "If using statevector simulator, operator should be numpy array.")

                        wavefunction = [result.data.statevector for result in results.results if
                                        getattr(getattr(result, "header", None), "name", "") == f"{idx}psi"]

                        assert len(wavefunction) == 1, f"Only one simulated circuit should match name {idx}psi."

                        mean, stdev = np.array(wavefunction[0]).T.conj() @ operator @ np.array(wavefunction[0]), 0.

                        zero_overlap = abs(wavefunction[0][0]) ** 2

                    # whereas shot based qasm_simulation runs different circuit for each pauli basis and uses
                    # methods TPBGroupedWeightedPauliOperator to recombine and calculate total expectation.
                    else:
                        assert type(operator) is TPBGroupedWeightedPauliOperator, (TypeError(
                            "If using shot based simulator, operator should be TPBGroupedWeightedPauliOperator"))

                        mean, stdev = operator.evaluate_with_result(
                            result=results, statevector_mode=self.quantum_instance.is_statevector,
                            circuit_name_prefix=str(idx + offset_idx))

                        zero_overlap = None
                        if return_zero_state_overlap:
                            if Pauli("Z" * self.n) in [b[0] for b in operator.basis]:
                                zzzz_counts = results.get_counts(str(idx + offset_idx) + "Z" * self.n)
                                zero_overlap = zzzz_counts['0' * self.n] / sum(zzzz_counts.values())

                    outputs.append((np.real(mean), np.real(stdev)))
                    zero_state_overlaps.append(zero_overlap)

            assert len(outputs) == len(parameter_sets), ValueError('Mismatch between input and output lengths.')

            if return_zero_state_overlap:
                return outputs, zero_state_overlaps

            return outputs

    def perform_repeated_measurements(self, x: np.ndarray, num_evaluations: int):
        """
        perform repeated calculation of cost function for same parameters. To be used for estimating effect of noise.
        """

        measured_energies = []
        predicted_variances = []

        repeats = self.evaluate_cost_function([x, ] * num_evaluations, self.h_operator)
        for (h, h_std) in repeats:
            measured_energies.append(h)
            predicted_variances.append(h_std ** 2)

        return measured_energies, predicted_variances

    def lanczos_mitigation(self, x: np.ndarray,
                           num_measurements: int = 100,
                           noise_extrap: str = 'cnots',
                           save_output: bool = True) \
            -> tuple[list[np.ndarray | Any], list[float | Any], list[float | Any], list[float | Any], list[float | Any],
            list[float | Any], list[float | Any]]:

        raise NotImplementedError("Lanczos mitigation is not included in the open access version of this code.")

    def subtract_uncoupled_noise(self,
                                 x: np.ndarray,
                                 x_uncoupled: Union[np.ndarray, None] = None,
                                 num_evaluations: int = 10) -> Tuple[float, float]:
        """
        Evaluate difference in cost function for given set of parameters x, and (optional) set of uncoupled parameters
        x_uncoupled. If x_uncoupled None, then zero parameters used. Different evaluations are interleved and submitted
        together such that noise level is as similar as possible between each param set. Returns difference between
        two expectation values and variance.
        """
        if x_uncoupled is None:
            x_uncoupled = [0, ] * len(x)
        assert len(x) == len(x_uncoupled), ValueError("x and x_uncoupled must be same length.")

        combined_parameters = list(chain(*zip(repeat(x, num_evaluations), repeat(x_uncoupled, num_evaluations))))

        h1_results_combined = self.evaluate_cost_function(combined_parameters, self.h_operator)
        h1_results_coupled = np.array(h1_results_combined[0::2])
        h1_results_uncoupled = np.array(h1_results_combined[1::2])
        print('h1 subtraction done')

        # h2_results_combined = self.evaluate_cost_function(combined_parameters, self.h2_operator)
        # h2_results_coupled = np.array(h2_results_combined[0::2])
        # h2_results_uncoupled = np.array(h2_results_combined[1::2])
        # print('h2 done')
        #
        # h3_results_combined = self.evaluate_cost_function(combined_parameters, self.h3_operator)
        # h3_results_coupled = np.array(h3_results_combined[0::2])
        # h3_results_uncoupled = np.array(h3_results_combined[1::2])
        # print('h3 done')
        #
        # np.save('full_subtraction_terms', {'h1': [h1_results_coupled, h1_results_uncoupled],
        #                                    'h2': [h2_results_coupled, h2_results_uncoupled],
        #                                    'h3': [h3_results_coupled, h3_results_uncoupled]})
        #
        h1_sub = [np.mean(h1_results_coupled[:, 0] - h1_results_uncoupled[:, 0]) + self.N,
                  (np.var(h1_results_coupled[:, 0]) + np.var(h1_results_uncoupled[:, 0]) / num_evaluations)]
        # h2_sub = [np.mean(h2_results_coupled[:, 0] - h2_results_uncoupled[:, 0]) + self.N ** 2,
        #           (np.var(h2_results_coupled[:, 0]) + np.var(h2_results_uncoupled[:, 0]) / num_evaluations)]
        # h3_sub = [np.mean(h3_results_coupled[:, 0] - h3_results_uncoupled[:, 0]) + self.N ** 3,
        #           (np.var(h3_results_coupled[:, 0]) + np.var(h3_results_uncoupled[:, 0]) / num_evaluations)]
        #
        # subtracted_lanczos = lanczos_algorithm(h1_sub, h2_sub, h3_sub)

        # np.save('full_subtraction_terms', {'h1': [h1_results_coupled, h1_results_uncoupled]})

        return h1_sub

    def name(self):
        pass

    def evaluate_state(self, x: np.ndarray) -> np.ndarray:
        """
        Calculates state for plotting
        """

        para_dic = {k: l for k, l in zip(self.all_params, x)}
        state_simulator = Aer.get_backend('statevector_simulator')
        qobj = assemble(self.qc_l.bind_parameters(para_dic), backend=state_simulator)
        job = state_simulator.run(qobj)
        state = job.result().get_statevector()
        state = np.exp(-1j * np.angle(state[0])) * state / np.sqrt(np.inner(state, np.conj(state)))

        return np.array(state)

    def do_measurement_calibration(self) -> CompleteMeasFitter:
        """
        Creates a measurement calibration fitter as described in
        https://qiskit.org/textbook/ch-quantum-hardware/measurement-error-mitigation.html and the Qiskit Ignis
        tutorial. This should be done once before all circuits since it is potentially expensive.
        :return A measurement calibration fitter. See qiskit's CompleteMeasFitter class.
        """

        print('Calculating measurement calibration')
        meas_calibs, state_labels = complete_meas_cal(qubit_list=[val for (key, val) in self.qubit_layout.items()],
                                                      qr=self.transpiled_l.qregs[0], circlabel='mcal')
        mcal_quantum_instance = copy(self.quantum_instance)
        mcal_quantum_instance.run_config.shots *= 20
        cal_results = mcal_quantum_instance.execute(meas_calibs, had_transpiled=True)
        print('Finished calculating measurement calibration')
        return CompleteMeasFitter(cal_results, state_labels, qubit_list=None, circlabel='mcal')
