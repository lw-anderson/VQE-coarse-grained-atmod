import time

import numpy as np
from qiskit.aqua.operators import WeightedPauliOperator, TPBGroupedWeightedPauliOperator
from qiskit.aqua.operators.legacy import op_converter
from qiskit.quantum_info import Pauli
from scipy import sparse
from scipy.sparse import linalg

from measure_commuting_operators import one_factorisation
from measure_commuting_operators.sorted_insertion import sorted_insertion
from operators.hamiltonian import get_harmonic_hamiltonian
from operators.calculate_pauli_coefficients import append_indices_and_coeffs_to_list, \
    calculate_pauli_coefficient_coupled_pair, calculate_pauli_coefficient_non_interaction
from simplectic_ground_state import hamiltonian_correlation_terms, simplectic_matrix, simplectic_ground_state_energy

# Copied and modified from NCoupledQHOFunc
def create_groups_for_coupling_terms(n, N, gammas, encoding='bin'):
    """
    Create groups of coefficients and indices for all coupling terms (for up to six oscillators). Using one
    factorisations of the coupling graphs terms acting on different oscillators are grouped together. There is no
    grouping of terms within each oscillator pair, this will be done by another method in the future.
    """
    pair_osc_coupled_indices, pair_osc_coupled_coeffs = calculate_pauli_coefficient_coupled_pair(
        int(2 * n / N), encoding)
    grouped_pairs = sorted_insertion(pair_osc_coupled_indices, pair_osc_coupled_coeffs, use=True)
    one_factorisation_groups = []

    gammas_matrix = gammas

    single_oscillator_identity_indices = (0,) * int(n / N)

    # Groups combining all interactions using 1-factorisations.
    for group in grouped_pairs:
        for covering in one_factorisation.k(N):

            one_factorisation_coeffs, one_factorisation_indices = [], []

            for (indices, coeff) in group:

                for (i, j) in covering:
                    coeff2 = coeff * gammas_matrix[i, j]
                    indices2 = single_oscillator_identity_indices * i \
                               + indices[:int(n / N)] \
                               + single_oscillator_identity_indices * (j - i - 1) \
                               + indices[int(n / N):] \
                               + single_oscillator_identity_indices * (N - j - 1)

                    one_factorisation_indices, one_factorisation_coeffs \
                        = append_indices_and_coeffs_to_list(indices2, coeff2, one_factorisation_indices,
                                                            one_factorisation_coeffs)

            one_factorisation_groups.append(list(zip(one_factorisation_indices, one_factorisation_coeffs)))

    # Groups split into the separate two oscillator interactions.
    separate_interaction_groups = []
    for covering in one_factorisation.k(N):
        for (i, j) in covering:
            if gammas_matrix[i, j] != 0.0:
                new_groups = []
                for group in grouped_pairs:
                    individual_coupling_indices, individual_coupling_coeffs = [], []
                    for (indices, coeff) in group:
                        coeff2 = coeff * gammas_matrix[i, j]
                        indices2 = single_oscillator_identity_indices * i \
                                   + indices[:int(n / N)] \
                                   + single_oscillator_identity_indices * (j - i - 1) \
                                   + indices[int(n / N):] \
                                   + single_oscillator_identity_indices * (N - j - 1)
                        individual_coupling_indices, individual_coupling_coeffs \
                            = append_indices_and_coeffs_to_list(indices2, coeff2, individual_coupling_indices,
                                                                individual_coupling_coeffs)

                    new_groups.append(list(zip(individual_coupling_indices, individual_coupling_coeffs)))

                separate_interaction_groups.append(new_groups)

    return one_factorisation_groups, separate_interaction_groups


def create_groups_for_non_interacting_terms(n, N, encoding='bin'):
    """
    Creates groups of coefficients and indices for the non interacting terms. All terms here act on different
    qubits and so will be returned as a single group.
    """

    one_osc_non_int_indices, one_osc_non_int_coeffs = calculate_pauli_coefficient_non_interaction(
        int(n / N), encoding)

    single_oscillator_identity_indices = (0,) * int(n / N)

    # Groups combining all oscillators together
    non_int_indices, non_int_coeffs = [], []
    for indices, coeff in zip(one_osc_non_int_indices, one_osc_non_int_coeffs):
        for i in range(N):
            indices1 = single_oscillator_identity_indices * i \
                       + indices \
                       + single_oscillator_identity_indices * (N - i - 1)
            non_int_indices, non_int_coeffs = append_indices_and_coeffs_to_list(indices1, coeff, non_int_indices,
                                                                                non_int_coeffs)
        combined_oscillator_groups = [list(zip(non_int_indices, non_int_coeffs))]

    # Groups separated into hamiltonian terms for each oscillator
    individual_oscillator_groups = []
    for i in range(N):
        non_int_indices, non_int_coeffs = [], []
        for indices, coeff in zip(one_osc_non_int_indices, one_osc_non_int_coeffs):
            indices1 = single_oscillator_identity_indices * i \
                       + indices \
                       + single_oscillator_identity_indices * (N - i - 1)
            non_int_indices, non_int_coeffs = append_indices_and_coeffs_to_list(indices1, coeff, non_int_indices,
                                                                                non_int_coeffs)
        individual_oscillator_groups.append([list(zip(non_int_indices, non_int_coeffs))])

    return combined_oscillator_groups, individual_oscillator_groups


num_oscillators = 2
qubits_per_oscillator = 2
gamma = 0.5

print('N  |  m  |  # terms  |   # groups   |   calc time (gen,sort)   '
      '|   E simp   |   E trunc   | calc time (simp,trunc)')
output = []

for qubits_per_oscillator in [1, 2, 3, 4]:
    for num_oscillators in [1, 2, 3, 4, 5]:

        t1 = time.time()

        gammas = np.array([gamma, ] * (num_oscillators * num_oscillators)).reshape((num_oscillators, num_oscillators)) \
                 - np.diag([gamma, ] * num_oscillators)

        non_int_groups_all_oscillators, non_int_groups_separate_oscillators \
            = create_groups_for_non_interacting_terms(qubits_per_oscillator * num_oscillators, num_oscillators)

        if num_oscillators == 1:
            coupling_groups_all_pairs, coupling_groups_separate_pairs = [], []
        else:
            coupling_groups_all_pairs, coupling_groups_separate_pairs \
                = create_groups_for_coupling_terms(qubits_per_oscillator * num_oscillators, num_oscillators, gammas)

        paulis = []
        for group in non_int_groups_all_oscillators + coupling_groups_all_pairs:
            for (indices, coeff) in group:
                pauli_string = ''
                pauli_options = ['I', 'X', 'Y', 'Z']
                for ind in tuple(reversed(indices)):
                    pauli_string += pauli_options[ind]
                paulis.append([coeff, Pauli(pauli_string)])

        t2 = time.time()

        expected_num_paulis = num_oscillators * (num_oscillators - 1) / 2 \
                              * ((2 ** qubits_per_oscillator) * qubits_per_oscillator / 2) ** 2 \
                              + num_oscillators * qubits_per_oscillator \
                              + 1

        h_operator = op_converter.to_tpb_grouped_weighted_pauli_operator(WeightedPauliOperator(paulis),
                                                                         TPBGroupedWeightedPauliOperator.sorted_grouping)
        num_terms = len(h_operator.paulis)
        num_groups = len(h_operator.basis)
        t3 = time.time()

        # Energy using simplectic representation
        two_point_corr_ham = hamiltonian_correlation_terms(num_oscillators, gammas / 2)
        simp_mat = simplectic_matrix(num_oscillators)
        simp_ground_state_energy = simplectic_ground_state_energy(simp_mat, two_point_corr_ham)
        t4 = time.time()

        # Energy using truncated fock bases

        trunc_ham = sparse.csr_matrix(
            get_harmonic_hamiltonian(num_oscillators, num_oscillators * qubits_per_oscillator, gammas, encoding='bin',
                                     numpy_array=False))

        eval, evec = linalg.eigsh(trunc_ham, k=1, which='SM')
        ev_list = [tup for tup in zip(eval, evec)]
        ev_list.sort(key=lambda tup: tup[0], reverse=False)
        trunc_ground_state_energy = ev_list[0][0]

        t5 = time.time()

        output.append((num_oscillators, qubits_per_oscillator, num_terms, num_groups, simp_ground_state_energy,
                       trunc_ground_state_energy))

        np.save(f'grouping_and_truncation_scaling/scaling_gamma_{gamma}', output)

        print(
            f'{num_oscillators}   |   {qubits_per_oscillator}   |   {num_terms}    |    {num_groups}   |    '
            f'{(t3 - t1):.3f} ({(t2 - t1):.3f},{(t3 - t2):.3f})   |'
            f'{simp_ground_state_energy:.4f}   |   {trunc_ground_state_energy:.4f}   |   '
            f'{(t5 - t3):.3f} ({(t4 - t3):.3f},{(t5 - t4):.3f})')
