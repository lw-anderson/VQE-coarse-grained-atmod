from typing import List, Union

import numpy as np
from graycode import gray_code_to_tc
from scipy.sparse import csr_matrix, kron
from sympy import sqrt, eye, diag
from sympy.matrices import zeros, Matrix
from sympy.physics.quantum import TensorProduct
from sympy.physics.secondquant import matrix_rep, B, VarBosonicBasis, Bd


def get_harmonic_hamiltonian(num_oscillators, num_qubits, gammas, encoding, numpy_array=False):
    """
    Get hamiltonian for multiple oscillators with arbitrary couplings given by gammas.
    """
    if num_qubits % num_oscillators != 0:
        raise ValueError('Number of qubits must be mutliple of number of oscillators.')
    if gammas.shape != (num_oscillators, num_oscillators):
        raise ValueError('Gammas must by NxN matrix for N oscillators.')
    m = int(num_qubits / num_oscillators)

    a = matrix_rep(B(0), VarBosonicBasis(2 ** m))
    aD = matrix_rep(Bd(0), VarBosonicBasis(2 ** m))
    X = (aD + a) / sqrt(2)
    Id = eye(2 ** m)
    N = diag(range(2 ** m), unpack=True)
    K = 2 * N + Id
    if encoding == 'bin':
        pass
    elif encoding == 'gray':
        a = binary_to_gray(a)
        aD = binary_to_gray(aD)
        X = binary_to_gray(X)
        Id = binary_to_gray(Id)
        N = binary_to_gray(N)
        K = binary_to_gray(K)
    else:
        raise ValueError('encoding should be either bin or gray.')

    K = csr_matrix(np.array(K, dtype=float))
    Id = csr_matrix(np.array(Id, dtype=float))
    X = csr_matrix(np.array(X, dtype=float))

    H = sparse_multi_tensor_product([K, ] + [Id, ] * (num_oscillators - 1))

    for i in range(1, num_oscillators):
        H += sparse_multi_tensor_product([Id, ] * i + [K, ] + [Id, ] * (num_oscillators - i - 1))

    for i in range(0, num_oscillators - 1):
        for j in range(i + 1, num_oscillators):
            H += sparse_multi_tensor_product(
                [Id, ] * i + [X, ] + [Id, ] * (j - i - 1) + [X, ] + [Id, ] * (num_oscillators - j - 1)
            ).multiply(gammas[i, j])
    if numpy_array:
        return H.toarray()
    else:
        return H


def get_extended_hamiltonian(num_oscillators: int, num_qubits: int, gammas: np.ndarray, cubic_prefactor: float = 0.0,
                             quartic_prefactor: float = 0.0, external_field: float = 0.0,
                             encoding: str = "bin", numpy_array: bool = True):
    """
    Get hamiltonian for multiple oscillators with arbitrary couplings given by gammas.

    Parameters
    ----------
    num_oscillators : (int) number of oscillators.
    num_qubits : (int) number of qubits total. Must be multiple of num_oscillators.
    gammas : (numpy array) NxN array of floats with off-diagonals specifying inter-oscillators coupling strengths.
    cubic_prefactor : (float) prefactor A to add Ax^3 potential term to oscillators.
    quartic_prefactor : (float) prefactor B to add Bx^4 potential term to oscillators.
    external_field : (floats) field strength E to add external field Ex to each oscillator.
    encoding : (str) encoding method for oscillators (either bin or gray).
    numpy_array : (bool) if True output numpy array, else sparse output.
    """
    if num_qubits % num_oscillators != 0:
        raise ValueError('Number of qubits must be mutliple of number of oscillators.')
    if gammas.shape != (num_oscillators, num_oscillators):
        raise ValueError('Gammas must by NxN matrix for N oscillators.')
    m = int(num_qubits / num_oscillators)

    a = matrix_rep(B(0), VarBosonicBasis(2 ** m))
    aD = matrix_rep(Bd(0), VarBosonicBasis(2 ** m))
    X = (aD + a) / sqrt(2)
    Id = eye(2 ** m)
    N = diag(range(2 ** m), unpack=True)
    K = 2 * N + Id
    if encoding == 'bin':
        pass
    elif encoding == 'gray':
        a = binary_to_gray(a)
        aD = binary_to_gray(aD)
        X = binary_to_gray(X)
        Id = binary_to_gray(Id)
        N = binary_to_gray(N)
        K = binary_to_gray(K)
    else:
        raise ValueError('encoding should be either bin or gray.')

    K = csr_matrix(np.array(K, dtype=float))
    Id = csr_matrix(np.array(Id, dtype=float))
    X = csr_matrix(np.array(X, dtype=float))
    Xsqu = X.dot(X)
    X3 = Xsqu.dot(X)
    X4 = X3.dot(X)

    H = get_harmonic_hamiltonian(num_oscillators, num_qubits, gammas, encoding, False)

    if cubic_prefactor:
        for i in range(num_oscillators):
            H += sparse_multi_tensor_product([Id, ] * i + [cubic_prefactor * X3, ] + [Id, ] * (num_oscillators - i - 1))

    if quartic_prefactor:
        for i in range(num_oscillators):
            H += sparse_multi_tensor_product(
                [Id, ] * i + [quartic_prefactor * X4, ] + [Id, ] * (num_oscillators - i - 1))

    if external_field:
        for i in range(num_oscillators):
            H += sparse_multi_tensor_product([Id, ] * i + [external_field * X, ] + [Id, ] * (num_oscillators - i - 1))

    if numpy_array:
        return H.toarray()
    else:
        return H


def single_oscillator_momentum_operator(n) -> Matrix:
    a = matrix_rep(B(0), VarBosonicBasis(2 ** n))
    aD = matrix_rep(Bd(0), VarBosonicBasis(2 ** n))

    p = 1j * (aD - a) / sqrt(2)

    return p


def multi_tensor_product(matrices: List[Matrix]) -> Matrix:
    """
    Calculate tensor product of arbitrary number of matrices.
    matrices: List of sympy matrices to tensor product together
    """
    prod = TensorProduct(matrices[0], matrices[1])
    for mat in matrices[2:]:
        prod = TensorProduct(prod, mat)
    return prod


def sparse_multi_tensor_product(matrices: List[csr_matrix]) -> csr_matrix:
    """
    Calculate tensor product of arbitrary number of ma.
    matrices: List of sympy matrices to tensor product together
    """
    if len(matrices) == 1:
        return matrices[0]
    prod = kron(matrices[0], matrices[1])
    if len(matrices) > 2:
        for mat in matrices[2:]:
            prod = kron(prod, mat)
    return csr_matrix(prod)


def binary_to_gray(matrix: Union[Matrix, np.ndarray]) -> Matrix:
    """
    Convert from binary to gray encoding.
    """
    new_mat = zeros(matrix.shape[0])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            new_mat[gray_code_to_tc(i), gray_code_to_tc(j)] = matrix[i, j]
    return new_mat
