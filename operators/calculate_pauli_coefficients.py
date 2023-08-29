import logging
import os
from itertools import product
from shutil import rmtree
from typing import Tuple, List

import numpy as np
from sympy import eye, diag, sqrt, re, trace
from sympy.physics import msigma
from sympy.physics.quantum import TensorProduct
from sympy.physics.secondquant import matrix_rep, B, VarBosonicBasis, Bd

from operators.hamiltonian import binary_to_gray


def calculate_pauli_coefficient_non_interaction(n: int, encoding: str, save_to_disk=False):
    """
    Pauli terms for non-interacting hamiltonian for single oscillator represented by n qubits.
    """
    dir_name = generate_saved_dir_name('non-int', n, encoding)

    if type(n) is not int or n < 0:
        raise (ValueError, "n must be positive int.")

    if os.path.exists(dir_name) and n != 1:
        non_zero_indices, non_zero_coefficients = load_coefficients(dir_name)

    else:
        Id = eye(2 ** n)
        N = diag(range(2 ** n), unpack=True)

        if encoding == 'bin':
            pass
        elif encoding == 'gray':
            Id = binary_to_gray(Id)
            N = binary_to_gray(N)
        else:
            raise ValueError('encoding should be either bin or gray.')

        H = (2 * N + Id)
        non_zero_indices, non_zero_coefficients = generate_and_save_coefficients(dir_name, H, n, save=save_to_disk)

    return non_zero_indices, non_zero_coefficients


def calculate_pauli_coefficient_coupled_pair(n: int, encoding: str, save_to_disk=False):
    """
    Pauli terms for coupling hamiltonian of two oscillators each represented by n/2 qubits.
    """
    dir_name = generate_saved_dir_name('coupled-pair', n, encoding)

    if type(n) is not int or n % 2 != 0:
        raise ValueError("n must be positive integer")

    if os.path.exists(dir_name) and n != 2:
        non_zero_indices, non_zero_coefficients = load_coefficients(dir_name)

    else:
        a = matrix_rep(B(0), VarBosonicBasis(2 ** int(n // 2)))
        aD = matrix_rep(Bd(0), VarBosonicBasis(2 ** int(n / 2)))
        X = (aD + a) / sqrt(2)

        if encoding == 'bin':
            pass
        elif encoding == 'gray':
            X = binary_to_gray(X)
        else:
            raise ValueError('encoding should be either bin or gray.')

        H = TensorProduct(X, X)
        non_zero_indices, non_zero_coefficients = generate_and_save_coefficients(dir_name, H, n, save=save_to_disk)

    return non_zero_indices, non_zero_coefficients


def append_indices_and_coeffs_to_list(indices: Tuple, coefficient: float, indices_list: List[Tuple],
                                      coefficient_list: List[float]):
    """
    Appends indices tuple and coefficient to a larger list. If the tuple is already present in original, it is not
    duplicated but instead the relevant coefficients are combined.
    """
    indices = tuple(int(i) for i in indices)
    indices_list = indices_list
    coefficient_list = coefficient_list

    for i, old_indices in enumerate(indices_list):
        if indices == old_indices:
            coefficient_list[i] = coefficient_list[i] + coefficient
            return indices_list, coefficient_list
    indices_list.append(indices)
    coefficient_list.append(coefficient)

    return indices_list, coefficient_list


def generate_saved_dir_name(cost_func, n, encoding):
    dir_name = os.path.join('pauli_coefficients_calculated', f'{cost_func}_{encoding}_n-{n}')
    return dir_name


def load_coefficients(dir_name):
    # logging.info(f'Coefficients have already been calculated, loading from {dir_name}.')
    non_zero_indices = [tuple(paulis) for paulis in np.loadtxt(os.path.join(dir_name, 'indices.txt'))]
    non_zero_coefficients = np.loadtxt(os.path.join(dir_name, 'coefficients.txt'))
    return non_zero_indices, non_zero_coefficients


def generate_and_save_coefficients(dir_name, H, n, save=False):
    # logging.info(f'Calculating coefficients and saving to {dir_name}.')

    non_zero_indices, non_zero_coefficients = calculate_non_zero_terms(H, n)

    if save:
        if os.path.exists(dir_name):
            rmtree(dir_name)
        os.makedirs(dir_name)
        np.savetxt(os.path.join(dir_name, 'indices.txt'), non_zero_indices)
        np.savetxt(os.path.join(dir_name, 'coefficients.txt'), non_zero_coefficients)

    return non_zero_indices, non_zero_coefficients


def calculate_coefficient(H: np.ndarray, pauli_indices: List[float]):
    """
    Given a list of indices [i,j,k,...] ∈ {0,1,2,3}^N labelling a basis operator within the N-qubit Pauli basis
        σ_i ⊗ σ_j ⊗ σ_k ⊗ ...,
    calculates the corresponding coefficient for the Hamiltonian by symbolically calculating the trace
        Tr[H (σ_i ⊗ σ_j ⊗ σ_k ⊗ ...)]
    where σ_i ∈ {I,X,Y,Z} are single qubit pauli operators (and the identity).
    """

    if H.shape[0] < 2 ** 8:
        pauli = [eye(2), msigma(1), msigma(2), msigma(3)]
        basis_state = pauli[pauli_indices[0]] / 2
        for index in pauli_indices[1:]:
            basis_state = TensorProduct(pauli[index], basis_state) / 2
        coeff = float(re(trace(H * basis_state)))

    else:
        pauli = [np.array([[1, 0], [0, 1]]), np.array([[0, 1], [1, 0]]),
                 np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])]
        basis_state = pauli[pauli_indices[0]] / 2
        for index in pauli_indices[1:]:
            basis_state = np.kron(pauli[index], basis_state) / 2
        coeff = np.trace(np.matmul(np.array(H).astype(np.float64), basis_state))

    if abs(coeff) < 1e-15:
        coeff = 0.0

    return coeff


def calculate_non_zero_terms(H: np.ndarray, n: int):
    """
    For all N qubit basis operators consisting of a tensor product of Pauli matrices, calculates the corresponding
    coefficient within the total (nonint + int) Hamiltonian. Returns a list containing the non-zero coefficients and a
    list containing the indices representing the corresponding basis states.
    """

    if H.shape != (2 ** n, 2 ** n):
        raise ValueError("H and num_qubits mismatch.")

    non_zero_indices = []
    non_zero_coefficients = []

    pauli_indices_list = product([0, 1, 2, 3], repeat=n)

    for i, pauli_indices in enumerate(pauli_indices_list):
        coeff = calculate_coefficient(H, pauli_indices)
        if coeff != 0:
            non_zero_indices.append(pauli_indices)
            if abs(np.imag(coeff)) > 1e-15:
                logging.warning(f'complex coeff calculated, value = {coeff}')
            non_zero_coefficients.append(np.real(coeff))

    return non_zero_indices, non_zero_coefficients
