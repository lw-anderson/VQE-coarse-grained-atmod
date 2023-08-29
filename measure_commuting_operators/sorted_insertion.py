from typing import List, Tuple


def sorted_insertion(pauli_indices: List, pauli_coefficients: List, use=False) -> List[List[Tuple[str, float]]]:
    """
    Implement sorted insertion algorithm from arXiv:1908.06942v2.
    Returns list of groups containing commuting pauli terms and corresponding coefficients.
    """

    if len(pauli_indices) != len(pauli_coefficients):
        raise ValueError("pauli_indices and operators should be lists of equal length.")

    ungrouped_pairs = [pair for pair in sort_tuple_coeff_indices_pair(pauli_indices, pauli_coefficients)]
    grouped_pairs = [[ungrouped_pairs[0]]]

    for (ind, coeff) in ungrouped_pairs[1:]:
        added = False
        if use:
            for group in grouped_pairs:
                if not added and all([all(tuple_indices_commute(ind, ind2)) for (ind2, coeff2) in group]):
                    group.append((ind, coeff))
                    added = True
                    break
            if not added:
                grouped_pairs.append([(ind, coeff)])
        else:
            grouped_pairs.append([(ind, coeff)])
    return grouped_pairs


def single_indices_commute(a, b) -> bool:
    """
    For integers a & b check if corresponding pauli operators commute.
    Indices a & b translate to pauli operators:
        a,b = 0 -> I (identity, commutes with all)
        a,b = 1 -> σ_x
        a,b = 2 -> σ_y
        a,b = 3 -> σ_z
    """
    return a == 0 or b == 0 or a == b


def tuple_indices_commute(a_tup: tuple, b_tup: tuple) -> List[bool]:
    """
    For pair list containing pauli indices, return list of bools corresponding to whether the pauli operators in each
    list commute.
    """
    return list(single_indices_commute(a, b) for a, b in zip(a_tup, b_tup))


def sort_tuple_coeff_indices_pair(pauli_indices: List, pauli_coefficients: List) -> List[Tuple[str, float]]:
    """
    Sort list of pauli indices and corresponding coefficients into pairs, ordered in decreasing coefficient value.
    """
    pairs = [(ind, coeff) for (ind, coeff) in zip(pauli_indices, pauli_coefficients)]
    pairs = sorted(pairs, key=lambda pair: abs(pair[1]), reverse=True)
    return pairs
