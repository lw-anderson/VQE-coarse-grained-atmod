import numpy as np


def shot_distribution_and_measurements(grouped_pairs, total_shots, use, all_group_variances=None):
    """
    For a list of groups of commuting operators, calculates the shots distributed to each group, and the pauli
    measurements that must be made for each group.
    Parameter all_group_exp_vals is list of all expectation values of all pauli operators for all groups. These are
    usually as a result of the previous optimisation step and are used to assign shots distribution. If None is given,
    will estimate expectation values as average values depending on the number of Pauli operators in a string.
    """

    measurements_and_weights = []
    total_weight = 0.0

    zero_indices_tuple = (0,) * len(grouped_pairs[0][0][0])

    if all_group_variances is None:
        all_group_variances = np.full(len(grouped_pairs), None)

    for group, variance in zip(grouped_pairs, all_group_variances):
        current_indices_to_measure = zero_indices_tuple
        if variance is None:
            variance = 0.
            for (indices, coeff) in group:
                qubits = sum((i != 0 for i in indices))
                if qubits == 0:  # For identity term variance is zero.
                    var_of_paulis = 0
                else:  # Estimate variance using expectation average over all hilbert space.
                    var_of_paulis = 1 - 1 / (2 ** qubits + 1)
                variance += abs(coeff) ** 2 * var_of_paulis

                current_indices_to_measure = tuple(map(lambda i, j: max(i, j), current_indices_to_measure, indices))

        measurements_and_weights.append((current_indices_to_measure, np.nan_to_num(np.sqrt(variance))))
        # TODO: Decide if correct behaviour for negative variance (currently will give zero)
        total_weight += np.nan_to_num(np.sqrt(variance))

    shots_and_measurements = []

    for (current_indices_to_measure, weight) in measurements_and_weights:
        if use:
            # TODO: Decide if regularisation for at least one shot is correct
            shots = max(int(total_shots * weight / total_weight), 1)
        else:
            shots = max(int(total_shots / len(grouped_pairs)), 1)

        shots_and_measurements.append((shots, current_indices_to_measure))

    return shots_and_measurements
