import logging
from abc import abstractmethod, ABC
from datetime import datetime
from typing import List, Tuple

import numpy as np
from qiskit import ClassicalRegister, assemble, Aer, IBMQ, transpile, QuantumRegister, QuantumCircuit
from qiskit.providers.aer.noise import NoiseModel

from measure_commuting_operators import one_factorisation
from measure_commuting_operators.sorted_insertion import sorted_insertion
from noise_mitigation.measurement_calibration import do_measurement_calibration
from operators.calculate_pauli_coefficients import calculate_pauli_coefficient_non_interaction, \
    calculate_pauli_coefficient_coupled_pair, append_indices_and_coeffs_to_list, calculate_non_zero_terms
from operators.hamiltonian import get_harmonic_hamiltonian, get_extended_hamiltonian
from quantum_ansatz.quantum_ansatz_factory import QuantumAnsatzFactory


class CostFunc(ABC):
    def __init__(self, num_oscillators, encoding, ansatz, num_qubits, depth, gammas, optimiser, shots, backend,
                 noise_model, allow_dynamic_shots, anharmonic={}, lanczos=False, meas_cal=False,
                 save_temp_evals=True):
        self.save_temp_evals = save_temp_evals
        if save_temp_evals:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.temp_outputs_fname = 'temp-evals ' + now + '.npy'
            self.temp_params_fname = 'temp-params ' + now + '.npy'

        if num_oscillators < 1 or num_oscillators > 6:
            raise ValueError("num_oscillators must be an integer between 2 and 6 (inclusive).")

        if num_qubits % num_oscillators != 0:
            raise ValueError("num_qubits must be a multiple of num_oscillators.")

        if backend == 'qasm_simulator' and noise_model:
            print('noise model = ', noise_model)
            if noise_model == 'load':
                self.noise_backend = np.load('noise_backend.npy', allow_pickle=True)[0]
            else:
                raise NotImplementedError("To IBMQ devicesthat require premium access, modify the provideo options"
                                          "below.")
                provider = None
                while provider is None:
                    try:
                        IBMQ.load_account()
                        provider = IBMQ.get_provider(hub='add-your-hub', group='add-your-group',
                                                     project='add-your-project')
                    except Exception as e:
                        logging.warning(e)
                        logging.warning('Unable to get IBMQ provider, retrying.')
                self.noise_backend = provider.get_backend(noise_model)
            self.noise_model = NoiseModel.from_backend(self.noise_backend)
            self.backend = Aer.get_backend(backend, noise_model=self.noise_model)
            np.save('noise_backend', [self.noise_model])
            if meas_cal:
                self.meas_calibration_fitter = do_measurement_calibration(self.noise_model, num_qubits)

        elif backend in ['statevector_simulator', 'qasm_simulator']:
            self.backend = Aer.get_backend(backend)

        else:
            self.backend = None
            while self.backend is None:
                raise NotImplementedError("To IBMQ devicesthat require premium access, modify the provideo options"
                                          "below.")
                try:
                    IBMQ.load_account()
                    provider = IBMQ.get_provider(hub='add-your-hub', group='add-your-group',
                                                 project='add-your-project')
                    self.backend = provider.get_backend(backend)
                except Exception as e:
                    logging.warning(e)
                    logging.warning('Unable to get IBMQ provider and backend, retrying.')
            if meas_cal:
                self.meas_calibration_fitter = do_measurement_calibration(NoiseModel.from_backend(self.backend),
                                                                          num_qubits)

        print('Backend = ', self.backend, 'Noise model =', noise_model)

        self.N = num_oscillators
        self.n = num_qubits
        self.d = depth
        self.shots = shots
        self.gammas = gammas
        self.encoding = encoding

        if anharmonic:
            self.hamiltonian = get_extended_hamiltonian(num_oscillators, num_qubits, gammas=gammas,
                                                        cubic_prefactor=anharmonic["cubic"],
                                                        quartic_prefactor=anharmonic["quartic"],
                                                        external_field=anharmonic["field"],
                                                        encoding=encoding)
        else:
            self.hamiltonian = get_harmonic_hamiltonian(num_oscillators, num_qubits, gammas, encoding)

        self.allow_dynamic_shots = allow_dynamic_shots
        self.lanczos = lanczos

        self.optimiser = optimiser

        self.grid = np.linspace(0, 1, 2 ** self.n, endpoint=False)

        # For saving progress using certain optimisers
        self.costfunc_values = []
        self.parameter_values = []

        self.analytic_minimum = None

        self.groups = {}
        self.shots_and_measurements = {}
        self.circuits = {}

        non_int_groups_all_oscillators, non_int_groups_separate_oscillators \
            = self.create_groups_for_non_interacting_terms()

        coupling_groups_all_pairs, coupling_groups_separate_pairs \
            = self.create_groups_for_coupling_terms()

        self.groups['h'] = non_int_groups_all_oscillators + coupling_groups_all_pairs
        separated_groups = coupling_groups_separate_pairs + [non_int_groups_all_oscillators]
        self.ansatz = QuantumAnsatzFactory(ansatz, self.n, self.d, self.N).get()

        self.qr = QuantumRegister(self.n)
        self.qc_l = QuantumCircuit(self.qr)
        self.qc_l.append(self.ansatz.circuit, self.qc_l.qubits)

        self.transpiled_l = transpile(self.qc_l, optimization_level=3, basis_gates=['id', 'rz', 'sx', 'x', 'cx'])
        decomposed_circ = self.transpiled_l.decompose().decompose().decompose()
        print('Circuit operators = ', decomposed_circ.count_ops())
        print('Numper of pauli terms = ', sum([len(group) for group in self.groups['h']]))
        print('Number of measurement groups = ', len(self.groups['h']))

        self.all_params = self.ansatz.params
        self.all_ind = set(list(range(len(self.all_params))))

    def __call__(self, x, shot_distribution=0):
        """
        Simulate circuit and return cost function in correct format for the optimiser

        shot_distribution: int labelling whether this is for the objective func (value 0) or one of the 2n evaluations
        (values 1, 2... 2n) needed for gradient (for n parameters). This is used for the non-markovian shot distribution
        method used for shot noise reduction.
        """

        if self.optimiser in {'midaco', 'nlopt'}:
            cost_function = self.evaluate_cost_function(x)[0]

            if self.optimiser == 'nlopt':
                return cost_function
            elif self.optimiser == 'midaco':
                return [cost_function], [0.0]

        elif self.optimiser in ['aqgd', 'aqgd-fin-diff', 'cobyla']:

            num_params = len(self.all_params)

            all_params = x.reshape(-1, num_params)
            output = self.evaluate_cost_function(all_params, "h1")

            self.costfunc_values.append(output[0])
            self.parameter_values.append(x.reshape(-1, num_params)[0])
            if self.save_temp_evals:
                np.save(self.temp_outputs_fname, self.costfunc_values, allow_pickle=True)
                np.save(self.temp_params_fname, self.parameter_values, allow_pickle=True)

            return [exp for (exp, stdev) in output]

        elif self.optimiser == 'adam':

            num_params = len(self.all_params)

            all_params = x.reshape(-1, num_params)
            output = self.evaluate_cost_function(all_params, "h1")

            self.costfunc_values.append(output[0])
            self.parameter_values.append(x.reshape(-1, num_params)[0])
            if self.save_temp_evals:
                np.save(self.temp_outputs_fname, self.costfunc_values, allow_pickle=True)
                np.save(self.temp_params_fname, self.parameter_values, allow_pickle=True)

            return output

        else:
            raise ValueError("Invalid optimiser type.")

    @abstractmethod
    def name(self):
        pass

    def create_groups_for_measurement(self, operator='h') -> (List, List):
        """
        Creates groups of measurements for the whole hamiltonian. Alternatively, create groups of measurements for H^2
        and H^3 for use in Lanczos error mitigation.
        """

        if operator == 'h':
            return self.create_groups_for_non_interacting_terms()[0] + self.create_groups_for_coupling_terms()[0]
        elif operator in ['h2', 'h3']:
            return self.create_groups_for_lanczos_operators(operator)
        else:
            raise ValueError('operator must be one of h, h2, h3.')

    def create_groups_for_non_interacting_terms(self) -> (List, List):
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

    def create_groups_for_coupling_terms(self) -> (List, List):
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

    def create_groups_for_lanczos_operators(self, operator: str) -> List[List[Tuple[str, float]]]:
        """
        Create grouped terms for H^2 and H^3 for use in lanczos error mitigation.
        """
        if operator == 'h2':
            lanczos_op_indices, lanczos_op_coeffs = calculate_non_zero_terms(
                np.linalg.matrix_power(self.hamiltonian, 2), self.n)

        elif operator == 'h3':
            lanczos_op_indices, lanczos_op_coeffs = calculate_non_zero_terms(
                np.linalg.matrix_power(self.hamiltonian, 2), self.n)
        else:
            raise ValueError('operator must be one of h2 or h3.')

        grouped_pairs = sorted_insertion(lanczos_op_indices, lanczos_op_coeffs, use=True)

        return grouped_pairs

    def create_circuits_for_groups(self, shots_and_measurements: List[Tuple[int, Tuple]]) -> List[QuantumCircuit]:
        """
        Generate list of parameterised circuits for coupling Hamiltonian. Each circuit contains measurements of
        Pauli operators required for each term within coupling Hamiltonian.
        """
        circuits = []

        for (shots, paulis) in shots_and_measurements[0]:

            if paulis == (0,) * self.n:  # Identity term, no measurement needed
                qc_pauli_meas = None

            else:
                qc_pauli_meas = self.transpiled_l.copy()
                # qc_pauli_meas.barrier(self.qr)
                cr = ClassicalRegister(self.n, name='c1')
                qc_pauli_meas.add_register(cr)

                for i in range(self.n):
                    if paulis[i] == 1:  # x measurement
                        qc_pauli_meas.h(self.qr[i])
                        qc_pauli_meas.measure(self.qr[i], cr[i])
                    elif paulis[i] == 2:  # y measurement
                        qc_pauli_meas.s(self.qr[i])
                        qc_pauli_meas.h(self.qr[i])
                        qc_pauli_meas.measure(self.qr[i], cr[i])
                    else:  # z measurement (or redundant measurement)
                        qc_pauli_meas.measure(self.qr[i], cr[i])
                qc_pauli_meas.barrier(self.qr)
            qc_pauli_meas = transpile(qc_pauli_meas, backend=self.backend, optimization_level=3,
                                      coupling_map=self.backend.configuration().coupling_map)

            if hasattr(self, 'noise_backend'):
                qc_pauli_meas = transpile(qc_pauli_meas, backend=self.noise_backend, optimization_level=3,
                                          coupling_map=self.noise_backend.configuration().coupling_map,
                                          initial_layout=list(range(self.n)))
            circuits.append(qc_pauli_meas)

        return circuits

    @abstractmethod
    def evaluate_cost_function(self, all_params, operator: str = "h1",
                               use_dynamic_shots: bool = False, update_dynamic_shots: bool = False) \
            -> List[Tuple[float]]:
        pass

    def evaluate_state(self, x: List[float]) -> np.ndarray:
        """
        Calculates state for plotting
        """

        para_dic = {k: l for k, l in zip(self.all_params, x)}
        state_simulator = Aer.get_backend('statevector_simulator')
        qobj = assemble(self.transpiled_l.bind_parameters(para_dic), backend=state_simulator)
        job = state_simulator.run(qobj)
        state = job.result().get_statevector()
        state = np.exp(-1j * np.angle(state[0])) * state / np.sqrt(np.inner(state, np.conj(state)))

        return np.array(state)
