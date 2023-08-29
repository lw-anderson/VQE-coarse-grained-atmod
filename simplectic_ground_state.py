import logging

import numpy as np

"""
Functions to calculate exact groundstate energy of systems of coupled Harmonic oscillators.
"""


def simplectic_matrix(n: int) -> np.ndarray:
    """
    Simplectic matrix for n oscillator system. Output is 2n x 2n matrix.
    """
    w = np.array([[0, 1], [-1, 0]])
    omega = np.kron(np.eye(n, dtype=int), w)
    return omega


def hamiltonian_correlation_terms(n: int, gamma_array: np.ndarray) -> np.ndarray:
    """
    Generate 2n x 2n Matrix corresponding to Hamiltonian in terms of correlation coefficients.
    """
    assert gamma_array.shape == (n, n), f'Shape of gamma array must be ({n},{n}), currently {gamma_array.shape}'
    ham = np.eye(2 * n)
    for i in range(n):
        for j in range(n):
            if i != j:
                ham[2 * i, 2 * j] = gamma_array[i, j]
    return ham


def three_oscillator_exact_ground_state_energy(gamma12: float, gamma23: float, gamma31: float) -> float:
    """
    Calculate ground state of three 1D harmonic oscillator.
    """
    logging.warning('Deprecated, use exact_ground_state_energy instead.', DeprecationWarning)
    gammas = np.array([[0, gamma12 / 2, gamma31 / 2], [gamma12 / 2, 0, gamma23 / 2], [gamma31 / 2, gamma23 / 2, 0]])
    hamiltonian = hamiltonian_correlation_terms(3, gammas)
    simpletic_matrix = simplectic_matrix(3)
    return simplectic_ground_state_energy(simpletic_matrix, hamiltonian)


def two_oscillator_exact_ground_state_energy(gamma: float) -> float:
    """
    Calculate ground state of 1D Harmonic oscillator pair.
    """
    logging.warning('Deprecated, use exact_ground_state_energy instead.', DeprecationWarning)
    gammas = np.array([[0, gamma / 2], [gamma / 2, 0]])
    hamiltonian = hamiltonian_correlation_terms(2, gammas)
    omega = simplectic_matrix(2)
    return simplectic_ground_state_energy(omega, hamiltonian)


def exact_ground_state_energy(gamma_matrix: np.ndarray) -> float:
    """
    Given matrix corresponding to coupling, calculate exact groundstate energy.
    """
    num_oscillators = gamma_matrix.shape[0]
    hamiltonian = hamiltonian_correlation_terms(num_oscillators, gamma_matrix / 2)
    omega = simplectic_matrix(num_oscillators)
    return simplectic_ground_state_energy(omega, hamiltonian)


def simplectic_ground_state_energy(simplectic_matrix: np.ndarray, hamiltonian: np.ndarray) -> float:
    assert simplectic_matrix.shape == hamiltonian.shape

    mat = 1.j * np.matmul(simplectic_matrix, hamiltonian)
    eigen_vals = np.linalg.eig(mat)[0]
    if any([abs(np.imag(eig)) > 1e-13 for eig in eigen_vals]):
        print('Non real eigen vals')
        print(eigen_vals)
        ground_state_energy = np.nan
    else:
        ground_state_energy = sum(np.abs(eigen_vals) / 2)
    return ground_state_energy
